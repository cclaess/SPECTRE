import os
import sys
import argparse
from itertools import chain
from functools import partial

import torch
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks.vision_language import SigLIP3D
from spectre.models.vits import VisionTransformer, FeatureVisionTransformer
from spectre.models.qwen_text_encoders import Qwen2Model, Qwen2Config
from spectre.ssl.transforms import SigLipTransform
from spectre.configs import default_config_clip
from spectre.utils.config import setup
from spectre.utils.models import update_momentum
from spectre.utils.dataloader import get_dataloader, extended_collate
from spectre.utils.masking import MaskingGenerator
from spectre.utils.scheduler import CosineWarmupScheduler, cosine_schedule



def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain CLIP")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/clip_default.yaml",
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
                "name": "clip-pretrain-" + cfg.model.architecture,
                "dir": os.path.join(cfg.train.output_dir, "logs"),
            },
        )

    # Initialize backbone
    if (
        cfg.model.image_encoder.architecture in models.__dict__ 
        and cfg.model.image_encoder.architecture.startswith("vit")
    ):
        vision_encoder = models.__dict__[cfg.model.image_encoder.architecture](
            num_classes=0,
            dynamic_img_size=True,
        )
        
    else:
        raise NotImplementedError(f"Model {cfg.model.image_encoder.architecture} not implemented.")
    

    if cfg.model.image_encoder.vit_pretrained:
        accelerator.print(f"Loading pretrained weights from {cfg.model.image_encoder.vit_pretrained}")
        vision_encoder.load_state_dict(
            torch.load(cfg.model.image_encoder.vit_pretrained, map_location="cpu"),
            strict=False,
        )
        accelerator.print("Pretrained weights loaded.")
    else:
        accelerator.print("No pretrained weights provided.")

    vision_feature_comb = FeatureVisionTransformer(
        patch_dim=cfg.model.feature_comb.patch_dim,
        num_patches=cfg.model.feature_comb.num_patches,
        depth=cfg.model.feature_comb.depth,
        heads=cfg.model.feature_comb.num_heads,
        embed_dim=cfg.model.feature_comb.embed_dim,
    )

    if cfg.model.feature_comb.feature_vit_pretrained:
        accelerator.print(f"Loading pretrained weights from {cfg.model.feature_comb.feature_vit_pretrained}")
        vision_feature_comb.load_state_dict(
            torch.load(cfg.model.feature_comb.feature_vit_pretrained, map_location="cpu"),
            strict=False,
        )
        accelerator.print("Pretrained weights loaded.")
    else:   
        accelerator.print("No pretrained weights provided.")
    
    model_config = Qwen2Config.from_pretrained(cfg.model.text_encoder.text_encoder_config)
    text_encoder = Qwen2Model(model_config)

    if cfg.model.text_encoder.text_encoder_pretrained:
        accelerator.print(f"Loading pretrained weights from {cfg.model.text_encoder.text_encoder_pretrained}")
        loaded = torch.load(cfg.model.text_encoder.text_encoder_pretrained, map_location="cpu")
        text_encoder.load_state_dict(loaded, strict=False)
        accelerator.print("Pretrained weights loaded.")
    else:
        accelerator.print("No pretrained weights provided. Attempting to download pretrained weights.")
        text_encoder = Qwen2Model.from_pretrained(cfg.model.text_encoder.text_encoder_config, trust_remote_code=True)
        accelerator.print("Pretrained weights downloaded.")

    # Initialize the SigLIP model
    model = SigLIP3D(vision_encoder, None, vision_feature_comb, None, text_encoder, None, embed_dim=768)

    # Get dataloader
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=True,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        transform=SigLipTransform(),
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
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
    model, data_loader, optimizer, lr_scheduler = accelerator.prepare(
            model, data_loader, optimizer, lr_scheduler,
        )
    
    # Keep unwrapped model for easier access to individual components
    unwrapped_model = accelerator.unwrap_model(model)

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(data_loader)

    # Start training
    global_step: int = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        for batch in data_loader:

            with accelerator.accumulate(model):

                # Update weight decay
                weight_decay = cosine_schedule(
                    global_step,
                    total_num_steps,
                    cfg.optim.weight_decay,
                    cfg.optim.weight_decay_end,
                )
                optimizer.param_groups[0]["weight_decay"] = weight_decay

                # Forward pass
                loss = unwrapped_model.forward(
                   batch['image'], batch['input_ids'], batch['attention_mask'], return_loss = True
                )


                # Backward pass
                accelerator.backward(loss)

                # Update model
                if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        chain(
                            unwrapped_model.student_backbone.parameters(), 
                            unwrapped_model.student_head_dino.parameters(),
                            unwrapped_model.student_head_ibot.parameters(),
                        ),
                        cfg.optim.clip_grad_norm
                    )

                optimizer.step()

                # Log loss, lr, and weight decay
                if global_step % cfg.train.log_freq == 0:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{cfg.optim.epochs}, "
                        f"Step {global_step + 1}/{total_num_steps}, "
                        f"Loss: {loss.item():8f}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}, "
                        f"Weight Decay: {weight_decay:.8f}, "
                        f"Momentum: {momentum:.8f}"
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "weight_decay": weight_decay,
                            "momentum": momentum,
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
    cfg = setup(args, default_config_clip)
    main(cfg)
