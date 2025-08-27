import os
import random
import argparse
from itertools import chain

import numpy as np
import torch
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import DINOv2
from spectre.ssl.losses import DINOLoss, KoLeoLoss, iBOTPatchLoss
from spectre.ssl.transforms import DINOTransform
from spectre.configs import default_config_dinov2
from spectre.utils import (
    setup,
    random_block_mask,
    update_momentum,
    get_dataloader,
    extended_collate_dino,
    load_state,
    save_state,
    cosine_schedule,
    cosine_warmup_schedule,
    linear_warmup_schedule,
)


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

    # Initialize backbone
    if (
        hasattr(models, cfg.model.architecture)
        and cfg.model.architecture.startswith("vit")
    ):
        backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
            dynamic_img_size=True,
        )
        embed_dim = backbone.embed_dim
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    # Get dataloader
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=False,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        use_gds=cfg.train.use_gds,
        transform=DINOTransform(
            num_local_views=cfg.train.num_local_views,
            dtype="float16" if cfg.train.load_fp16 else "float32",
            use_gds=cfg.train.use_gds,
        ),
        fraction=cfg.train.data_fraction,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
        collate_fn=extended_collate_dino,
        drop_last=cfg.train.drop_last,
        persistent_workers=cfg.train.persistent_workers,
        use_thread=cfg.train.use_thread,
    )

    # Initialize DINO model
    model = DINOv2(
        backbone,
        input_dim=embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
        output_dim=cfg.model.output_dim,
        ibot_seperate_head=cfg.model.ibot_seperate_head,
        student_drop_path_rate=cfg.model.student_drop_path_rate,
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
    criterion_ibot = iBOTPatchLoss(
        output_dim=cfg.model.output_dim,
        teacher_temp=cfg.model.teacher_temp,
        student_temp=cfg.model.student_temp,
        center_momentum=cfg.model.center_momentum,
    )
    criterion_koleo = KoLeoLoss()

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )

    # Prepare model, data, and optimizer for training
    model, data_loader, criterion_dino, criterion_ibot, criterion_koleo, \
        optimizer = accelerator.prepare(
            model, data_loader, criterion_dino,
            criterion_ibot, criterion_koleo, optimizer,
        )
    
    # Keep unwrapped model for easier access to individual components
    unwrapped_model = accelerator.unwrap_model(model)

    # Load checkpoint if specified
    if cfg.train.resume_ckp:
        start_epoch = load_state(
            os.path.join(cfg.train.output_dir, "checkpoint.pt"),
            model=unwrapped_model,
            optimizer=optimizer, 
            criterion_dino=criterion_dino, 
            criterion_koleo=criterion_koleo,
            criterion_ibot=criterion_ibot,
        )
        if start_epoch > 0:
            accelerator.print(f"Resuming training from epoch {start_epoch}.")

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(data_loader)
    warmup_num_steps = cfg.optim.warmup_epochs * len(data_loader)
    warmup_teacher_temp_num_steps = cfg.optim.warmup_teacher_temp_epochs * len(data_loader)

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

                # Update momentum
                momentum = cosine_schedule(
                    global_step,
                    total_num_steps,
                    cfg.model.momentum_teacher,
                    cfg.model.momentum_teacher_end,
                )
                update_momentum(unwrapped_model.student_backbone.vit, unwrapped_model.teacher_backbone.vit, momentum)
                update_momentum(unwrapped_model.student_head_dino, unwrapped_model.teacher_head_dino, momentum)
                if cfg.model.ibot_seperate_head:
                    update_momentum(unwrapped_model.student_head_ibot, unwrapped_model.teacher_head_ibot, momentum)

                # Update weight decay
                weight_decay = cosine_schedule(
                    global_step,
                    total_num_steps,
                    cfg.optim.weight_decay,
                    cfg.optim.weight_decay_end,
                )
                optimizer.param_groups[0]["weight_decay"] = weight_decay

                global_views = batch["global_views"].as_tensor()
                local_views = batch["local_views"].as_tensor()

                # Masking
                B = global_views.shape[0]
                sequence_length = unwrapped_model.teacher_backbone.vit.patch_embed.sequence_length
                num_prefix_tokens = unwrapped_model.teacher_backbone.num_prefix_tokens
                mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

                # Mask patches except for the cls token
                H, W, D = unwrapped_model.teacher_backbone.vit.patch_embed.grid_size
                assert H * W * D == sequence_length - num_prefix_tokens, \
                    "Grid size does not match sequence length."
                block_mask = random_block_mask(size=(B, H, W, D), device=mask.device, seed=cfg.train.seed)
                mask[:, num_prefix_tokens:] = block_mask.flatten(start_dim=1)

                # Forward pass
                with torch.no_grad():
                    teacher_cls_out, teacher_masked_out = unwrapped_model.forward_teacher(
                        global_views=global_views,
                        mask=mask,
                    )
                student_cls_out, student_masked_out = unwrapped_model.forward_student(
                    global_views=global_views,
                    local_views=local_views,
                    mask=mask,
                )

                # Calculate the loss
                teacher_temp = linear_warmup_schedule(
                    step=global_step,
                    warmup_steps=warmup_teacher_temp_num_steps,
                    start_value=cfg.model.warmup_teacher_temp,
                    end_value=cfg.model.teacher_temp,
                )
                dino_loss = criterion_dino(
                    teacher_out=teacher_cls_out.chunk(2, dim=0),
                    student_out=student_cls_out.chunk(2 + cfg.model.num_local_views, dim=0),
                    teacher_temp=teacher_temp,
                )
                ibot_loss = criterion_ibot(
                    teacher_out=teacher_masked_out,
                    student_out=student_masked_out,
                    mask=block_mask,
                    teacher_temp=teacher_temp,
                )
                koleo_loss = sum(
                    criterion_koleo(p) for p in student_cls_out[:2].chunk(2, dim=0)
                )

                loss = cfg.model.dino_loss_weight * dino_loss + \
                    cfg.model.ibot_loss_weight * ibot_loss + \
                    cfg.model.koleo_loss_weight * koleo_loss

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
                        f"LR: {lr:.8f}, "
                        f"Weight Decay: {weight_decay:.8f}, "
                        f"Momentum: {momentum:.8f}"
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "dino_loss": dino_loss.item(),
                            "koleo_loss": koleo_loss.item(),
                            "ibot_loss": ibot_loss.item(),
                            "epoch": epoch,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "momentum": momentum,
                        },
                        step=global_step,
                    )
                
                # Zero gradients
                optimizer.zero_grad()

                # Update global step
                global_step += 1

        # Save checkpoint
        if accelerator.is_main_process:
            save_state(
                os.path.join(cfg.train.output_dir, "checkpoint.pt"),
                epoch=epoch + 1,
                model=unwrapped_model,
                optimizer=optimizer,
                criterion_dino=criterion_dino,
                criterion_koleo=criterion_koleo,
                criterion_ibot=criterion_ibot,
                torch_random_state=torch.random.get_rng_state(),
                numpy_random_state=tuple(np.random.get_state()),
                random_random_state=random.getstate(),
            )
            if (epoch + 1) % cfg.train.saveckp_freq == 0:
                save_state(
                    os.path.join(cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}.pt"),
                    epoch=epoch + 1,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    criterion_dino=criterion_dino,
                    criterion_koleo=criterion_koleo,
                    criterion_ibot=criterion_ibot,
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
    cfg, accelerator = setup(args, default_config_dinov2)
    main(cfg, accelerator)
