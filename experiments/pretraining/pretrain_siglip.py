import os
import random
import argparse
from itertools import chain
from functools import partial

import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from accelerate import Accelerator
from safetensors import safe_open
from transformers import (
    XLMRobertaModel, 
    XLMRobertaTokenizerFast, 
    XLMRobertaConfig,
    Qwen2TokenizerFast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3Config,
)

import spectre.models as models
from spectre.ssl.frameworks import SigLIP
from spectre.ssl.losses import SigLIPLoss, CLIPLoss
from spectre.ssl.transforms import SigLIPTransform
from spectre.configs import default_config_siglip
from spectre.utils import (
    setup,
    get_global_size,
    get_dataloader,
    extended_collate_siglip,
    add_lora_adapters,
    load_state,
    save_state,
    cosine_warmup_schedule,
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
        tokenizer=XLMRobertaTokenizerFast.from_pretrained(
            cfg.model.text_tokenizer,
        ),
        # tokenizer=Qwen2TokenizerFast.from_pretrained(
        #     cfg.model.text_tokenizer,
        # ),
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
        ),
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=cfg.train.drop_last,
        persistent_workers=cfg.train.persistent_workers,
    )

    # Initialize backbone
    if (
        hasattr(models, cfg.model.architecture) 
        and cfg.model.architecture.startswith("vit")
    ):
        image_backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
            global_pool='',
        )
        image_backbone_embed_dim = image_backbone.embed_dim
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    image_feature_comb = models.FeatureVisionTransformer(
        num_patches=36,
        patch_dim=image_backbone_embed_dim * 2,  # cls token + avg pooling (C. Jose et al. 2024)
        num_classes=0,
        global_pool='',
        embed_dim=cfg.model.feature_comb_embed_dim,
        depth=cfg.model.feature_comb_num_layers,
        num_heads=cfg.model.feature_comb_num_heads,
    )
    
    # Initialize text backbone
    # TODO: add support for other text backbones
    # AutoModel is not yet compatible with newest Pytorch Docker image
    config = {
        "architectures": [
            "XLMRobertaModel"
        ],
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "classifier_dropout": None,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 8194,
        "model_type": "xlm-roberta",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "output_past": True,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "torch_dtype": "float32",
        "transformers_version": "4.52.3",
        "type_vocab_size": 1,
        "use_cache": True,
        "vocab_size": 250002
    }
    
    text_backbone = XLMRobertaModel(XLMRobertaConfig.from_dict(config))

    text_pretrained_weights = torch.load(cfg.model.text_encoder_weights, map_location="cpu")
    msg = text_backbone.load_state_dict(
        text_pretrained_weights, strict=True
    )
    accelerator.print(f"Pretrained weights of text encoder loaded with msg: {msg}")
    # config = {
    #     "_attn_implementation_autoset": True,
    #     "architectures": [
    #         "Qwen3ForCausalLM"
    #     ],
    #     "attention_bias": False,
    #     "attention_dropout": 0.0,
    #     "bos_token_id": 151643,
    #     "eos_token_id": 151643,
    #     "head_dim": 128,
    #     "hidden_act": "silu",
    #     "hidden_size": 1024,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 3072,
    #     "max_position_embeddings": 32768,
    #     "max_window_layers": 28,
    #     "model_type": "qwen3",
    #     "num_attention_heads": 16,
    #     "num_hidden_layers": 28,
    #     "num_key_value_heads": 8,
    #     "rms_norm_eps": 1e-06,
    #     "rope_scaling": None,
    #     "rope_theta": 1000000,
    #     "sliding_window": None,
    #     "tie_word_embeddings": True,
    #     "torch_dtype": "float32",
    #     "use_cache": True,
    #     "use_sliding_window": False,
    #     "vocab_size": 151669
    # }
    # text_backbone = Qwen3Model(Qwen3Config.from_dict(config))

    # # Load pretrained weights for text backbone
    # text_pretrained_weights = {}
    # with safe_open(cfg.model.text_encoder_weights, framework="pt", device="cpu") as f:
    #     for key in f.keys():
    #         # Skip the keys that are not part of the model
    #         if "lm_head" in key or "model.embed_tokens" in key:
    #             continue
    #         text_pretrained_weights[key] = f.get_tensor(key)
    # msg = text_backbone.load_state_dict(
    #     text_pretrained_weights, strict=True
    # )
    # accelerator.print(f"Pretrained weights of text encoder loaded with msg: {msg}")
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

    # Initialize the SigLIP model
    model = SigLIP(
        image_backbone=image_backbone,
        text_backbone=text_backbone,
        image_feature_comb=image_feature_comb,
        image_embed_dim=image_feature_comb.embed_dim * 2,
        text_embed_dim=text_backbone_embed_dim,
        projection_dim=cfg.model.projection_dim,
        backbone_is_class_token=False,  # backbone returns all tokens
        backbone_combine_features=True,  # use cls token + avg pooling (C. Jose et al. 2024)
        feature_comb_is_class_token=False,  # feature combiner returns all tokens
        feature_comb_combine_features=True,  # use cls token + avg pooling (C. Jose et al. 2024)
    )

    # Intialize criterion
    # criterion = SigLIPLoss(
    #     learnable_t=cfg.model.learnable_t,
    #     learnable_b=cfg.model.learnable_b,
    #     normalize=cfg.model.normalize,
    #     init_t=cfg.model.init_t,
    #     init_b=cfg.model.init_b,
    # )
    criterion = CLIPLoss(
        normalize=cfg.model.normalize,
    )

    # Initialize optimizer
    optimizer = AdamW(
        chain(
            model.parameters(),
            criterion.parameters(),
        ) if cfg.model.learnable_t or cfg.model.learnable_b else \
            model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )

    # Prepare model, data, and optimizer for training
    model, data_loader, criterion, optimizer = accelerator.prepare(
        model, data_loader, criterion, optimizer,
    )

    # Keep unwrapped model for easier access to individual components
    unwrapped_model = accelerator.unwrap_model(model)

    # Load checkpoint if specified
    if cfg.train.resume_ckp:
        start_epoch = load_state(
            os.path.join(cfg.train.output_dir, "checkpoint.pt"),
            model=unwrapped_model,
            optimizer=optimizer, 
            criterion=criterion,
        )
    else:
        start_epoch: int = 0
    if start_epoch > 0:
        start_epoch += 1
        accelerator.print(f"Resuming training from epoch {start_epoch}.")

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(data_loader)
    warmup_num_steps = cfg.optim.warmup_epochs * len(data_loader)

    # Start training
    global_step: int = start_epoch * len(data_loader)
    for epoch in range(start_epoch, cfg.optim.epochs):
        model.train()
        for batch_num, batch in enumerate(data_loader):

            with accelerator.accumulate(model):

                # Update learning rate
                lr = cosine_warmup_schedule(
                    global_step,
                    max_steps=total_num_steps,
                    start_value=cfg.optim.lr,
                    end_value=cfg.optim.min_lr,
                    warmup_steps=warmup_num_steps,
                    warmup_start_value=0.0,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward pass
                image_embeddings, text_embeddings = model(
                    batch['image'], batch['input_ids'], batch['attention_mask']
                )

                # Get outputs fromn all devices
                image_embeddings = accelerator.gather(image_embeddings)
                text_embeddings = accelerator.gather(text_embeddings)

                loss = criterion(image_embeddings, text_embeddings)

                # Backward pass
                accelerator.backward(loss)

                # Set gradients of image and text encoders to zero if freezing backbone
                if epoch < cfg.optim.freeze_backbone_epochs:
                    for name, param in model.named_parameters():
                        if "image_backbone" in name or "text_backbone" in name:
                            if param.requires_grad:
                                param.grad = None
                
                unwrapped_model.image_projection.cancel_last_layer_gradients(epoch)
                unwrapped_model.text_projection.cancel_last_layer_gradients(epoch)

                # Update model
                if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm)

                optimizer.step()

                # Log loss, lr, and weight decay
                if global_step % cfg.train.log_freq == 0:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{cfg.optim.epochs}, "
                        f"Step {global_step + 1}/{total_num_steps}, "
                        f"Loss: {loss.item():8f}, "
                        f"LR: {lr:.8f}, "
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            # "pos_loss": details["pos_loss"] / get_global_size(),
                            # "neg_loss": details["neg_loss"] / get_global_size(),
                            "epoch": epoch,
                            "lr": lr,
                        },
                        step=global_step,
                    )
                
                # Zero gradients
                optimizer.zero_grad()

                # Update global step
                global_step += 1

        if accelerator.is_main_process:
            save_state(
                os.path.join(cfg.train.output_dir, "checkpoint.pt"),
                epoch=epoch,
                model=unwrapped_model,
                optimizer=optimizer,
                criterion=criterion,
                torch_random_state=torch.random.get_rng_state(),
                numpy_random_state=tuple(np.random.get_state()),
                random_random_state=random.getstate(),
            )
            if (epoch + 1) % cfg.train.saveckp_freq == 0:
                save_state(
                    os.path.join(cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}.pt"),
                    epoch=epoch,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    criterion=criterion,
                    torch_random_state=torch.random.get_rng_state(),
                    numpy_random_state=tuple(np.random.get_state()),
                    random_random_state=random.getstate(),
                )
        accelerator.wait_for_everyone()
    
    # Make sure the trackers are finished before exiting
    accelerator.end_training()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg, accelerator = setup(args, default_config_siglip)
    main(cfg, accelerator)
