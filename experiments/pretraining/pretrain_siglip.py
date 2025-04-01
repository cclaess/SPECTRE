import os
import argparse
from functools import partial

import torch.nn as nn
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import SigLIP
from spectre.ssl.losses import SigLIPLoss
from spectre.ssl.transforms import SigLIPTransform
from spectre.configs import default_config_siglip
from spectre.utils.config import setup
from spectre.utils.dataloader import get_dataloader
from spectre.utils.scheduler import CosineWarmupScheduler



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
        "opts",
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


def main(cfg):
    """
    Main function to run pretraining.

    Args:
        cfg: Configuration object containing all hyperparameters and settings.
    """
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        log_with="wandb" if cfg.train.log_wandb else None,
    )

    # Print config
    accelerator.print(cfg)

    # Initialize wandb
    if cfg.train.log_wandb:
        accelerator.init_trackers(
            project_name="spectre",
            config={k: v for d in cfg.values() for k, v in d.items()},
            init_kwargs={
                "name": "siglip-pretrain-" + cfg.model.architecture,
                "dir": os.path.join(cfg.train.output_dir, "logs"),
            },
        )
    
    # Get dataloader
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=True,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        transform=SigLIPTransform(),
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    # Initialize backbone
    if (
        cfg.model.architecture in models.__dict__ 
        and cfg.model.architecture.startswith("vit")
    ):
        image_backbone = models.__dict__[cfg.model.architecture](
            num_classes=0,
        )
        backbone_embed_dim = image_backbone.embed_dim
    elif (
        cfg.model.architecture in models.__dict__
        and cfg.model.architecture.startswith("resnet")
        or cfg.model.architecture.startswith("resnext")
    ):
        image_backbone = models.__dict__[cfg.model.architecture](
            num_classes=0,
            norm_layer=partial(nn.BatchNorm3d, track_running_stats=False),
        )
        backbone_embed_dim = image_backbone.num_features
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    image_feature_comb = models.FeatureVisionTransformer(
        patch_dim=backbone_embed_dim,
        embed_dim=cfg.model.feature_comb_embed_dim,
        num_patches=36,
        depth=cfg.model.feature_comb_num_layers,
        heads=cfg.model.feature_comb_num_heads,
    )
    
    text_config = models.Qwen2Config.from_pretrained(cfg.model.text_encoder_config)
    text_backbone = models.Qwen2Model(config=text_config)

    # Initialize the SigLIP model
    model = SigLIP(
        image_backbone=image_backbone,
        text_backbone=text_backbone,
        image_feature_comb=image_feature_comb,
        image_embed_dim=image_feature_comb.embed_dim,
        text_embed_dim=text_config.hidden_size,
        projection_dim=cfg
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

    # Initialize learning rate scheduler
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.optim.warmup_epochs * len(data_loader),
        max_epochs=cfg.optim.epochs * len(data_loader),
        start_value=cfg.optim.lr,
        end_value=cfg.optim.min_lr,
    )

    # Prepare model, data, and optimizer for training
    model, data_loader, criterion, optimizer, lr_scheduler = accelerator.prepare(
        model, data_loader, criterion, optimizer, lr_scheduler,
    )

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(data_loader)

    # Start training
    global_step: int = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        for batch in data_loader:

            with accelerator.accumulate(model):

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
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}, "
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )
                
                # Zero gradients
                optimizer.zero_grad()

                # Update learning rate
                lr_scheduler.step()

                # Update global step
                global_step += 1

        if (epoch + 1) % cfg.train.saveckp_freq == 0 or (epoch + 1) == cfg.optim.epochs:
            accelerator.save_model(
                model,
                os.path.join(
                    cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}"
                ),
            )
    
    # Make sure the trackers are finished before exiting
    accelerator.end_training()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = setup(args, default_config_siglip)
    main(cfg)
