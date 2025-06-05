import os
import random
import time
import argparse
from itertools import chain
from functools import partial

import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import DINO
from spectre.ssl.losses import DINOLoss
from spectre.ssl.transforms import DINOTransform
from spectre.configs import default_config_dino
from spectre.utils.config import setup
from spectre.utils.models import update_momentum
from spectre.utils.dataloader import get_dataloader
from spectre.utils.collate import extended_collate_dino
from spectre.utils.checkpointing import load_state, save_state
from spectre.utils.scheduler import cosine_schedule, cosine_warmup_schedule


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain DINO")
    parser.add_argument(
        "--config_file",
        type=str,
        default="spectre/configs/dino_default.yaml",
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
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=False,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        use_gds=cfg.train.use_gds,
        transform=DINOTransform(
            dtype="float16" if cfg.train.load_fp16 else "float32",
        ),
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
        collate_fn=extended_collate_dino,
        drop_last=cfg.train.drop_last,
        persistent_workers=cfg.train.persistent_workers,
    )

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
    elif (
        hasattr(models, cfg.model.architecture)
        and cfg.model.architecture.startswith("resnet")
        or cfg.model.architecture.startswith("resnext")
    ):
        backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
            norm_layer=partial(nn.BatchNorm3d, track_running_stats=False),
        )
        embed_dim = backbone.num_features
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    # Initialize DINO model
    model = DINO(
        backbone,
        input_dim=embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
        output_dim=cfg.model.output_dim,
    )

    # Initialize criterion
    criterion = DINOLoss(
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
        epoch_start_time = time.time()
        model.train()
        for batch in data_loader:
            step_start_time = time.time()
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
                update_momentum(unwrapped_model.student_backbone, unwrapped_model.teacher_backbone, momentum)
                update_momentum(unwrapped_model.student_head, unwrapped_model.teacher_head, momentum)

                # Update weight decay
                weight_decay = cosine_schedule(
                    global_step,
                    total_num_steps,
                    cfg.optim.weight_decay,
                    cfg.optim.weight_decay_end,
                )
                optimizer.param_groups[0]["weight_decay"] = weight_decay

                # Forward pass
                teacher_cls_tokens_global = unwrapped_model.forward_teacher(
                    global_crops=batch["global_crops"]
                )
                student_cls_tokens_global, student_cls_tokens_local = model(
                    global_crops=batch["global_crops"], 
                    local_crops=batch["local_crops"]
                )

                loss = criterion(
                    teacher_cls_tokens_global.chunk(2, dim=0),
                    student_cls_tokens_global.chunk(2, dim=0) + student_cls_tokens_local.chunk(8, dim=0),
                    epoch=epoch,
                )

                # Backward pass
                accelerator.backward(loss)

                # Update model
                if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        chain(
                            unwrapped_model.student_backbone.parameters(), 
                            unwrapped_model.student_head.parameters(),
                        ),
                        cfg.optim.clip_grad_norm
                    )

                unwrapped_model.student_head.cancel_last_layer_gradients(epoch)
                optimizer.step()
                
                # Log loss, lr, and weight decay
                if global_step % cfg.train.log_freq == 0:
                    step_time= time.time() - step_start_time
                    epoch_time = time.time() - epoch_start_time
                    accelerator.print(
                        f"Epoch {epoch + 1}/{cfg.optim.epochs}, "
                        f"Step {global_step + 1}/{total_num_steps}, "
                        f"Loss: {loss.item():8f}, "
                        f"LR: {lr:.8f}, "
                        f"Weight Decay: {weight_decay:.8f}, "
                        f"Momentum: {momentum:.8f}, "
                        f"Step Time: {step_time:.4f}s, "
                        f"Epoch Time: {epoch_time:.4f}s"
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "epoch": epoch,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "momentum": momentum,
                            "step_time": step_time,
                            "epoch_time": epoch_time,
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
    cfg, accelerator = setup(args, default_config_dino)
    main(cfg, accelerator)
