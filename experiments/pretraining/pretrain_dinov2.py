import os
import argparse
from itertools import chain
from functools import partial

import torch
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import DINOv2
from spectre.ssl.losses import DINOLoss, KoLeoLoss, iBOTPatchLoss
from spectre.ssl.transforms import DINOTransform
from spectre.configs import default_config_dinov2
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
    parser = argparse.ArgumentParser(description="Pretrain DINOv2")
    parser.add_argument(
        "--config_file",
        type=str,
        default="spectre/configs/dinov2_default.yaml",
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
                "name": "dinov2-pretrain-" + cfg.model.architecture,
                "dir": os.path.join(cfg.train.output_dir, "logs"),
            },
        )

    # Initialize backbone
    if (
        cfg.model.architecture in models.__dict__ 
        and cfg.model.architecture.startswith("vit")
    ):
        backbone = models.__dict__[cfg.model.architecture](
            num_classes=0,
            dynamic_img_size=True,
        )
        embed_dim = backbone.embed_dim
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    # Get dataloader
    collate_fn = partial(
        extended_collate, 
        mask_ratio=(cfg.model.mask_ratio_min, cfg.model.mask_ratio_max), 
        mask_probability=cfg.model.mask_probability, 
        n_tokens=backbone.patch_embed.num_patches,
        mask_generator=MaskingGenerator(
            input_size=backbone.patch_embed.grid_size,
            max_num_patches=0.5 * backbone.patch_embed.num_patches,
        )
    )
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=False,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        transform=DINOTransform(),
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Initialize DINO model
    model = DINOv2(
        backbone,
        input_dim=embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
        output_dim=cfg.model.output_dim,
    )

    # Initialize criterion
    criterion_dino = DINOLoss(
        output_dim=cfg.model.output_dim,
        warmup_teacher_temp=cfg.model.warmup_teacher_temp,
        teacher_temp=cfg.model.teacher_temp,
        warmup_teacher_temp_epochs=cfg.model.warmup_teacher_temp_epochs,
        student_temp=cfg.model.student_temp,
        center_momentum=cfg.model.center_momentum,
    )
    criterion_koleo = KoLeoLoss()
    criterion_ibot = iBOTPatchLoss(
        output_dim=cfg.model.output_dim,
        warmup_teacher_temp=cfg.model.warmup_teacher_temp,
        teacher_temp=cfg.model.teacher_temp,
        warmup_teacher_temp_epochs=cfg.model.warmup_teacher_temp_epochs,
        student_temp=cfg.model.student_temp,
        center_momentum=cfg.model.center_momentum,
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
    model, data_loader, criterion_dino, criterion_koleo, criterion_ibot, \
        optimizer, lr_scheduler = accelerator.prepare(
            model, data_loader, criterion_dino, criterion_koleo,
            criterion_ibot, optimizer, lr_scheduler,
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

                # Update momentum
                momentum = cosine_schedule(
                    global_step,
                    total_num_steps,
                    cfg.model.momentum_teacher,
                    cfg.model.momentum_teacher_end,
                )
                update_momentum(unwrapped_model.student_backbone.vit, unwrapped_model.teacher_backbone, momentum)
                update_momentum(unwrapped_model.student_head_dino, unwrapped_model.teacher_head_dino, momentum)
                update_momentum(unwrapped_model.student_head_ibot, unwrapped_model.teacher_head_ibot, momentum)

                # Update weight decay
                weight_decay = cosine_schedule(
                    global_step,
                    total_num_steps,
                    cfg.optim.weight_decay,
                    cfg.optim.weight_decay_end,
                )
                optimizer.param_groups[0]["weight_decay"] = weight_decay

                # Forward pass
                teacher_cls_tokens_global, teacher_patch_tokens_global = unwrapped_model.forward_teacher(
                    global_crops=batch["global_crops"].as_tensor(), 
                    mask_indices=batch["mask_indices"], 
                    upperbound=batch["upperbound"]
                )
                student_cls_tokens_global, student_patch_tokens_global, student_cls_tokens_local = model(
                    global_crops=batch["global_crops"].as_tensor(), 
                    local_crops=batch["local_crops"].as_tensor(), 
                    masks=torch.cat([
                        torch.zeros(batch["masks"].shape[0], 1, dtype=torch.bool, device=batch["masks"].device), 
                        batch["masks"]
                    ], dim=1),  # Add cls token to mask here, not sure where to do this yet ...
                    mask_indices=batch["mask_indices"], 
                    upperbound=batch["upperbound"]
                )

                dino_loss = criterion_dino(
                    teacher_cls_tokens_global.chunk(2, dim=0),
                    student_cls_tokens_global.chunk(2, dim=0) + student_cls_tokens_local.chunk(8, dim=0),
                    epoch=epoch,
                )

                koleo_loss = sum(
                    criterion_koleo(p) for p in student_cls_tokens_global.chunk(2, dim=0)
                )

                ibot_loss = criterion_ibot.forward_masked(
                    teacher_patch_tokens_global,
                    student_patch_tokens_global,
                    mask=batch["masks"],
                    epoch=epoch,
                    masks_weight=batch["masks_weight"],
                )

                loss = cfg.model.dino_loss_weight * dino_loss + \
                    cfg.model.koleo_loss_weight * koleo_loss + \
                    cfg.model.ibot_loss_weight * ibot_loss

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

                unwrapped_model.student_head_dino.cancel_last_layer_gradients(epoch)
                unwrapped_model.student_head_ibot.cancel_last_layer_gradients(epoch)
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
    cfg = setup(args, default_config_dinov2)
    main(cfg)
