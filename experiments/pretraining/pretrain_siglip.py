import os
import random
import argparse
from functools import partial

import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer

import spectre.models as models
from spectre.ssl.frameworks import SigLIP
from spectre.ssl.losses import SigLIPLoss
from spectre.ssl.transforms import SigLIPTransform
from spectre.configs import default_config_siglip
from spectre.utils.config import setup
from spectre.utils.dataloader import get_dataloader
from spectre.utils.collate import extended_collate_siglip
from spectre.utils.checkpointing import load_state, save_state
from spectre.utils.scheduler import cosine_warmup_schedule


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
        tokenizer=AutoTokenizer.from_pretrained(
            cfg.model.text_tokenizer, trust_remote_code=True
        ),
    )
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=True,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
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
        )
        image_backbone_embed_dim = image_backbone.embed_dim
    elif (
        hasattr(models, cfg.model.architecture)
        and cfg.model.architecture.startswith("resnet")
        or cfg.model.architecture.startswith("resnext")
    ):
        image_backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
            norm_layer=partial(nn.BatchNorm3d, track_running_stats=False),
        )
        image_backbone_embed_dim = image_backbone.num_features
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    image_feature_comb = models.FeatureVisionTransformer(
        patch_dim=image_backbone_embed_dim,
        embed_dim=cfg.model.feature_comb_embed_dim,
        num_patches=36,
        depth=cfg.model.feature_comb_num_layers,
        heads=cfg.model.feature_comb_num_heads,
    )
    
    # Initialize text backbone
    text_backbone = AutoModel.from_pretrained(
        pretrained_model_name_or_path=cfg.model.text_encoder, 
        trust_remote_code=True
    )
    text_backbone_embed_dim = text_backbone.config.hidden_size

    # Initialize the SigLIP model
    model = SigLIP(
        image_backbone=image_backbone,
        text_backbone=text_backbone,
        image_feature_comb=image_feature_comb,
        image_embed_dim=image_feature_comb.embed_dim,
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
    optimizer = AdamW(
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

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(data_loader)
    warmup_num_steps = cfg.optim.warmup_epochs * len(data_loader)

    # Start training
    global_step: int = start_epoch * len(data_loader)
    for epoch in range(start_epoch, cfg.optim.epochs):
        model.train()
        for batch in data_loader:

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

                loss = criterion(image_embeddings, text_embeddings)

                # Backward pass
                accelerator.backward(loss)

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
            accelerator.save_state(
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
