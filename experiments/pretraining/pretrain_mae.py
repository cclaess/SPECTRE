import os
import argparse
from itertools import chain

from torch.nn import MSELoss
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import MAE
from spectre.ssl.transforms import MAETransform
from spectre.configs import default_config_mae
from spectre.utils.config import setup
from spectre.utils.dataloader import get_dataloader
from spectre.utils.scheduler import CosineWarmupScheduler


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain DINO")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/mae_default.yaml",
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
        log_with="wandb" if cfg.train.log_wandb else None,
    )

    # Print config
    accelerator.print(cfg)

    # Initialize wandb
    if cfg.train.log_wandb:
        accelerator.init_trackers(
            project_name="spectre",
            # config=vars(cfg),
            init_kwargs={
                "name": "mae-pretrain-" + cfg.model.architecture,
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
        transform=MAETransform(),
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
        )
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    # Initialize DINO model
    model = MAE(
        backbone,
        mask_ratio=cfg.model.mask_ratio,
        decoder_dim=cfg.model.decoder_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
    )

    # Initialize criterion
    criterion = MSELoss()

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        weight_decay=cfg.optim.weight_decay,
    )

    # calculate number of steps for training
    num_steps = cfg.optim.epochs * len(data_loader) // accelerator.num_processes
    num_warmup_steps = (
        cfg.optim.warmup_epochs * len(data_loader) // accelerator.num_processes
    )

    # Initialize learning rate scheduler
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=num_warmup_steps,
        max_epochs=num_steps,
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
    unwrapped_model = accelerator.unwrap_model(model)

    # Start training
    global_step: int = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        for batch in data_loader:

            optimizer.zero_grad()

            # Update learning rate
            lr_scheduler.step()

            # Forward pass
            outputs, targets = model(batch["image"])
            loss = criterion(outputs, targets)

            # Backward pass
            accelerator.backward(loss)

            # Update model
            if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unwrapped_model, cfg.optim.clip_grad_norm)

            optimizer.step()

            # Log loss, lr, and weight decay
            if global_step % cfg.train.log_freq == 0:
                accelerator.print(
                    f"Epoch {epoch+1}/{cfg.optim.epochs}, "
                    f"Step {global_step}/{num_steps}, "
                    f"Loss: {loss.item():8f}, "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.8f}"
                )
                accelerator.log(
                    {
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            # Update global step
            global_step += 1

        if (epoch + 1) % cfg.train.saveckp_freq == 0:
            accelerator.save_model(
                model,
                os.path.join(
                    cfg.train.output_dir, f"checkpoint_epoch={epoch + 1:04}.pt"
                ),
            )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = setup(args, default_config_mae)
    main(cfg)
