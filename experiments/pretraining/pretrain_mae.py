import os
import random
import argparse

import torch
import numpy as np
from torch.nn import MSELoss
from torch.optim import AdamW
from accelerate import Accelerator

import spectre.models as models
from spectre.ssl.frameworks import MAE
from spectre.ssl.transforms import MAETransform
from spectre.configs import default_config_mae
from spectre.utils.config import setup
from spectre.utils.dataloader import get_dataloader
from spectre.utils.scheduler import cosine_warmup_schedule
from spectre.utils.checkpointing import load_state, save_state


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain MAE")
    parser.add_argument(
        "--config_file",
        type=str,
        default="spectre/configs/mae_default.yaml",
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
    # Initialize accelerator
    accelerator.print(cfg)

    # Get dataloader
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=False,
        include_labels=False,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        transform=MAETransform(
            dtype="float16" if cfg.train_load_fp16 else "float32",
        ),
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
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
                outputs, targets = model(batch["image"])
                loss = criterion(outputs, targets)

                # Backward pass
                accelerator.backward(loss)

                # Update model
                if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unwrapped_model.parameters(), cfg.optim.clip_grad_norm)

                optimizer.step()

                # Log loss, lr, and weight decay
                if global_step % cfg.train.log_freq == 0:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{cfg.optim.epochs}, "
                        f"Step {global_step + 1}/{total_num_steps}, "
                        f"Loss: {loss.item():8f}, "
                        f"LR: {lr:.8f}"
                    )
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr": lr,
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
    cfg, accelerator = setup(args, default_config_mae)
    main(cfg, accelerator)
