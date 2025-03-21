import os
import argparse
from itertools import chain

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
from spectre.utils.scheduler import CosineWarmupScheduler, cosine_schedule


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain DINO")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/dino_default.yaml",
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
                "name": "dino-pretrain-" + cfg.model.architecture,
                "dir": os.path.join(cfg.train.output_dir, "logs"),
            },
        )

    # Get dataloader
    data_loader = get_dataloader(
        cfg.train.dataset,
        cfg.train.dataset_path,
        include_reports=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        transform=DINOTransform(),
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
        backbone = models.__dict__[cfg.model.architecture](
            num_classes=0,
            dynamic_img_size=True,
        )
        embed_dim = backbone.embed_dim
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
        model,
        data_loader,
        criterion,
        optimizer,
        lr_scheduler,
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

            optimizer.zero_grad()

            # Update learning rate
            lr_scheduler.step()

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

            with accelerator.accumulate(model):

                # Forward pass
                teacher_outputs = [unwrapped_model.forward_teacher(view) for view in batch["global_crops"]]
                student_outputs = [model(view) for view in batch["global_crops"] + batch["local_crops"]]

                loss = criterion(teacher_outputs, student_outputs, epoch=epoch)

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

            # Update global step
            global_step += 1

        if (epoch + 1) % cfg.train.saveckp_freq == 0 or (epoch + 1) == cfg.optim.epochs:
            accelerator.save_model(
                model,
                os.path.join(
                    cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}"
                ),
            )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = setup(args, default_config_dino)
    main(cfg)
