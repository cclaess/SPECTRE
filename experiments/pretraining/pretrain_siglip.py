import os
import time
import argparse
from itertools import chain
from functools import partial

import torch.nn as nn
from torch.optim import AdamW
from accelerate import Accelerator
from safetensors import safe_open
from transformers import (
    Qwen2TokenizerFast,
    Qwen3Model,
    Qwen3Config,
)

import spectre.models as models
from spectre.ssl.frameworks import SigLIP
from spectre.ssl.losses import SigLIPLoss
from spectre.ssl.transforms import SigLIPTransform
from spectre.configs import default_config_siglip
from spectre.utils import (
    setup,
    get_dataloader,
    extended_collate_siglip,
    add_lora_adapters,
    load_state,
    save_state,
    cosine_warmup_schedule,
    get_param_groups_with_decay,
)


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain SigLIP")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/siglip_default.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="command line arguments to override config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="output directory to save checkpoints and logs",
    )
    return parser


def main(cfg, accelerator: Accelerator):
    """
    Main function to run pretraining.

    Args:
        cfg: Configuration object containing all hyperparameters and settings.
        accelerator: Accelerator object for distributed training.
    """
    # Print config
    accelerator.print(cfg)
    
    # Get dataloader
    collate_fn = partial(
        extended_collate_siglip,
        tokenizer=Qwen2TokenizerFast.from_pretrained(
            cfg.model.text_tokenizer,
        ),
    )
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=True,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        use_gds=cfg.train.use_gds,
        transform=SigLIPTransform(
            dtype="float16" if cfg.train.load_fp16 else "float32",
            use_gds=cfg.train.use_gds,
        ),
        fraction=cfg.train.data_fraction,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=cfg.train.drop_last,
        persistent_workers=cfg.train.persistent_workers,
        use_thread=cfg.train.use_thread,
    )
    accelerator.print(f"Number of samples in dataloader: {len(data_loader.dataset)}")

    # Initialize backbone
    if (
        hasattr(models, cfg.model.architecture) 
        and cfg.model.architecture.startswith("vit")
    ):
        image_backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
            global_pool='',
            pos_embed="rope",
            rope_kwargs={
                "base": 1000.0,  # works for most 3D models
                "rescale_coords": 2.0,  # s in [0.5, 2.0]
            },
            init_values=cfg.model.layer_scale_init_value,
        )
        image_backbone_embed_dim = image_backbone.embed_dim
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")
    
    if cfg.optim.freeze_backbone_epochs < 0:  # -1 means fully freeze backbone
        for n, p in image_backbone.named_parameters():
            p.requires_grad = False  # freeze image backbone
        image_backbone.eval()

    if cfg.model.use_feature_comb:
        image_feature_comb = models.FeatureVisionTransformer(
            patch_dim=image_backbone_embed_dim * 2,  # cls token + avg pooling (C. Jose et al. 2024)
            num_classes=0,
            global_pool='',
            pos_embed="rope",
            rope_kwargs={
                "base": 100.0,  # less patches -> slower rotation
                "rescale_coords": 2.0,  # s in [0.5, 2.0]
            },
            init_values=cfg.model.layer_scale_init_value,
            embed_dim=cfg.model.feature_comb_embed_dim,
            depth=cfg.model.feature_comb_num_layers,
            num_heads=cfg.model.feature_comb_num_heads,
        )
    else:
        image_feature_comb = None
    
    # Initialize text backbone
    # TODO: add support for other text backbones
    # AutoModel is not yet compatible with newest Pytorch Docker image
    config = {
        "_attn_implementation_autoset": True,
        "architectures": [
            "Qwen3ForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 32768,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151669
    }
    text_backbone = Qwen3Model(Qwen3Config.from_dict(config))

    # Load pretrained weights for text backbone
    text_pretrained_weights = {}
    with safe_open(cfg.model.text_encoder_weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            # Skip the keys that are not part of the model
            if "lm_head" in key or "model.embed_tokens" in key:
                continue
            text_pretrained_weights[key] = f.get_tensor(key)
    msg = text_backbone.load_state_dict(
        text_pretrained_weights, strict=True
    )
    accelerator.print(f"Pretrained weights of text encoder loaded with msg: {msg}")
    text_backbone_embed_dim = text_backbone.config.hidden_size

    # Add LoRA adapters to text backbone if specified
    if cfg.model.use_lora and cfg.model.lora_r > 0:
        add_lora_adapters(
            text_backbone,
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_keywords=cfg.model.lora_target_keywords,
        )
        for n, p in text_backbone.named_parameters():
            p.requires_grad = ('lora_' in n)
        accelerator.print(
            f"LoRA adapters added to text backbone. Trainable parameters: "
            f"{sum(p.numel() for p in text_backbone.parameters() if p.requires_grad):,d} / "
            f"{sum(p.numel() for p in text_backbone.parameters()):,d}."
        )
    else:
        for n, p in text_backbone.named_parameters():
            p.requires_grad = False  # freeze text backbone
            text_backbone.eval()
            accelerator.print(
                "No LoRA adapters added to text backbone. All parameters are frozen."
            )

    # Initialize the SigLIP model
    if cfg.model.use_feature_comb:
        image_embed_dim = cfg.model.feature_comb_embed_dim * 2  # use cls token + avg pooling (C. Jose et al. 2024)
    else:
        image_embed_dim = image_backbone_embed_dim * 2
    model = SigLIP(
        image_backbone=image_backbone,
        text_backbone=text_backbone,
        image_feature_comb=image_feature_comb,
        image_embed_dim=image_embed_dim,
        text_embed_dim=text_backbone_embed_dim,
        projection_dim=cfg.model.projection_dim,
    )

    # Intialize criterion
    criterion = SigLIPLoss(
        learnable_t=cfg.model.learnable_t,
        learnable_b=cfg.model.learnable_b,
        normalize=cfg.model.normalize,
        init_t=cfg.model.init_t,
        init_b=cfg.model.init_b,
    )

    # Initialize optimizer
    param_groups = get_param_groups_with_decay(
        model,
        llrd_factor=cfg.optim.llrd_factor,
        patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult,
        lora_lr_factor=1.0,  # use base lr for LoRA parameters
    )
    param_groups += get_param_groups_with_decay(
        criterion,
    )

    optimizer = AdamW(
        param_groups,
        lr=cfg.optim.lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        weight_decay=cfg.optim.weight_decay,
    )

    # Prepare model, data, and optimizer for training
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, data_loader, criterion, optimizer = accelerator.prepare(
        model, data_loader, criterion, optimizer,
    )

    # Keep unwrapped model for easier access to individual components
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_criterion = accelerator.unwrap_model(criterion)

    # Load checkpoint if specified
    if cfg.train.resume_ckp:
        start_epoch = load_state(
            os.path.join(cfg.train.output_dir, "checkpoint.pt"),
            model=unwrapped_model,
            optimizer=optimizer, 
            criterion=criterion,
        )
        if start_epoch > 0:
            accelerator.print(f"Resuming training from epoch {start_epoch}.")

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(data_loader)
    warmup_num_steps = cfg.optim.warmup_epochs * len(data_loader)

    # Start training
    global_step: int = start_epoch * len(data_loader)
    t0 = time.time()
    for epoch in range(start_epoch, cfg.optim.epochs):

        # Set epoch for shuffling
        if hasattr(data_loader, "set_epoch"):
            data_loader.set_epoch(epoch)  # accelerate will call sampler internally

        for batch in data_loader:
            with accelerator.accumulate(model):

                # Update learning rate and weight decay
                lr = cosine_warmup_schedule(
                    global_step,
                    max_steps=total_num_steps,
                    start_value=cfg.optim.lr,
                    end_value=cfg.optim.min_lr,
                    warmup_steps=warmup_num_steps,
                    warmup_start_value=0.0,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * param_group.get("lr_mult", 1.0)
                    param_group["weight_decay"] = cfg.optim.weight_decay * param_group.get("wd_mult", 1.0)

                # Forward pass
                image_embeddings, text_embeddings = model(
                    images=batch["image"], 
                    text_tokens=batch['input_ids'],
                    image_grid_size=(3, 3, 4),  # hardcoded for 384x384x256 with 128x128x64 patches
                    attention_mask=batch['attention_mask'],
                )

                loss, details = criterion(
                    image_embeddings, 
                    text_embeddings,
                    return_details=True,
                )

                # Backward pass
                accelerator.backward(loss)

                # Set gradients of backbone to zero if specified
                # This is useful for freezing the backbone during the initial epochs
                if 0 < cfg.optim.freeze_backbone_epochs > epoch:
                    for n, p in model.named_parameters():
                        if "backbone_image" in n:
                            if p.requires_grad:
                                p.grad = None
                
                unwrapped_model.projection_image.cancel_last_layer_gradients(epoch)
                unwrapped_model.projection_text.cancel_last_layer_gradients(epoch)

                # Update model
                if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm)

                optimizer.step()

                # Log loss, lr, and weight decay
                step_time = time.time() - t0
                t0 = time.time()
                if global_step % cfg.train.log_freq == 0:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{cfg.optim.epochs}, "
                        f"Step {global_step + 1}/{total_num_steps}, "
                        f"Loss: {loss.item():8f}, "
                        f"LR: {lr:.8f}, "
                        f"Step Time: {step_time:.2f} sec"
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "pos_loss": details["pos_loss"],
                            "neg_loss": details["neg_loss"],
                            "temperature": unwrapped_criterion.t.item(),
                            "bias": unwrapped_criterion.b.item(),
                            "epoch": epoch,
                            "lr": lr,
                            "step_time": step_time,
                        },
                        step=global_step,
                    )
                
                if global_step % cfg.train.log_grad_freq == 0:
                    # Collect gradients
                    gradients = {}
                    for n, p in chain(model.named_parameters(), criterion.named_parameters()):
                        if p.requires_grad:
                            if p.grad is not None:
                                gradients[n] = p.grad.abs().mean().item()  # mean absolute grad
                            else:
                                gradients[n] = float("nan")  # param has no grad this step

                    # Log gradients to wandb
                    accelerator.log({
                        f"gradients/{n}": v for n, v in gradients.items()
                    }, step=global_step)
                
                # Zero gradients
                optimizer.zero_grad()

                # Update global step
                global_step += 1

        save_state(
            os.path.join(cfg.train.output_dir, "checkpoint.pt"),
            epoch=epoch + 1,
            model=unwrapped_model,
            optimizer=optimizer,
            criterion=criterion,
        )
        if (epoch + 1) % cfg.train.saveckp_freq == 0:
            save_state(
                os.path.join(cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}.pt"),
                epoch=epoch + 1,
                model=unwrapped_model,
                optimizer=optimizer,
                criterion=criterion,
            )
        accelerator.wait_for_everyone()
    
    # Make sure the trackers are finished before exiting
    accelerator.end_training()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg, accelerator = setup(args, default_config_siglip)
    main(cfg, accelerator)
