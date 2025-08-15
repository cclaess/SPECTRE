import os
import argparse

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from monai.data import  DataLoader
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, 
    LoadImaged, 
    CastToTyped,
    EnsureChannelFirstd, 
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    RandSpatialCropSamplesd,
    SpatialPadd,
    GridPatchd,
)

import spectre.data as data
import spectre.models as models
from spectre.configs import load_config
from spectre.losses import MaskClassificationLoss
from spectre.data.total_segmentator import LABEL_GROUPS
from spectre.utils import (
    setup,
    cosine_warmup_schedule,
    compute_backbone_lr_multipliers,
)


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Train Total Segmentator")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/classification_default.yaml",
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


def main(cfg, accelerator: Accelerator):

    # Print config
    accelerator.print(cfg)

    # Define transforms
    labels = [lbl for group in list(cfg.train.label_groups) for lbl in LABEL_GROUPS[group.lower()]]
    train_transform = Compose([
        LoadImaged(keys=["image"] + labels),
        CastToTyped(keys=labels, dtype=torch.uint8),
        EnsureChannelFirstd(keys=["image"] + labels, channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image"] + labels, axcodes="RAS"),
        Spacingd(
            keys=["image"] + labels,
            pixdim=(0.75, 0.75, 1.5),
            mode=["bilinear"] + ["nearest"] * len(labels),
        ),
        SpatialPadd(
            keys=["image"] + labels,
            spatial_size=(128, 128, 64),
        ),  # will only pad the smaller images
        RandSpatialCropSamplesd(
            keys=["image"] + labels,
            roi_size=(128, 128, 64),
            num_samples=12,
            random_size=False,
        ),
    ])

    val_transform = Compose([
        LoadImaged(keys=["image"] + labels),
        CastToTyped(keys=labels, dtype=torch.uint8),
        EnsureChannelFirstd(keys=["image"] + labels, channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image"] + labels, axcodes="RAS"),
        Spacingd(
            keys=["image"] + labels,
            pixdim=(0.75, 0.75, 1.5),
            mode=["bilinear"] + ["nearest"] * len(labels),
        ),
        SpatialPadd(
            keys=["image"] + labels,
            spatial_size=(128, 128, 64),
        ),  # will only pad the smaller images
        GridPatchd(
            keys=["image"] + labels,
            patch_size=(128, 128, 64),
        ),
    ])

    # Get dataloader
    data_kwargs = {
        "data_dir": cfg.train.data_dir,
        "include_labels": True,
        "label_groups": cfg.train.label_groups,
    }
    if cfg.train.cache_dataset:
        data_kwargs["cache_dir"] = cfg.train.cache_dir
    train_dataset = getattr(
        data, 
        "TotalSegmentatorDataset" if not cfg.train.cache_dataset \
            else "TotalSegmentatorCacheDataset")(
                **data_kwargs,
                transform=train_transform,
                subset="train",
            )
    val_dataset = getattr(
        data,
        "TotalSegmentatorDataset" if not cfg.train.cache_dataset \
            else "TotalSegmentatorCacheDataset")(
                **data_kwargs,
                transform=val_transform,
                subset="val",
            )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        drop_last=cfg.train.drop_last,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=1,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        drop_last=cfg.train.drop_last,
        shuffle=False,
    )
    
    # Initialize model (only support ViT models with EoMT for now)
    if (
        cfg.model.architecture in models.__dict__ 
        and cfg.model.architecture.startswith("vit")
    ):
        backbone = models.__dict__[cfg.model.architecture](
            global_pool='',  # Keep all tokens for EoMT
            num_classes=0,  # No classification head for EoMT
        )
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")
    
    # Load pretrained weights and freeze backbone if specified
    if cfg.model.pretrained_weights is not None:
        msg = backbone.load_state_dict(torch.load(cfg.model.pretrained_weights), strict=False)
        accelerator.print(f"Pretrained weights loaded with message: {msg}")
        if cfg.model.linear_only:
            for param in backbone.parameters():
                    param.requires_grad = False
    
    model = models.EoMT(
        backbone=backbone,
        num_classes=5,
        num_q=cfg.model.num_queries,
        num_blocks=cfg.model.num_blocks,
        masked_attn_enabled=cfg.model.masked_attn_enabled,
    )

    # Initialize criterion
    criterion = MaskClassificationLoss(
        num_labels=len(labels),
        num_points=cfg.model.num_points,
        oversample_ratio=cfg.model.oversample_ratio,
        importance_sample_ratio=cfg.model.importance_sample_ratio,
        mask_coefficient=cfg.model.mask_coefficient,
        dice_coefficient=cfg.model.dice_coefficient,
        class_coefficient=cfg.model.class_coefficient,
        no_object_coefficient=cfg.model.no_object_coefficient,
    )

    # Initialize optimizer
    param_to_mult = compute_backbone_lr_multipliers(model, cfg.optim.llrd)
    mult_to_params = {}
    for n, p in model.named_parameters():
        mult_to_params.setdefault(param_to_mult.get(n, 1.0), []).append(p)

    optimizer = AdamW(
        [{
            "params": params, 
            "lr": cfg.optim.lr * mult, 
            "multiplier": mult
        } for mult, params in mult_to_params.items()],
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )

    # Initialize Dice metric
    dice = DiceMetric(
        include_background=False,
        num_classes=len(labels),
    )

    # Prepare model, data, and optimizer for training
    model, train_dataloader, val_dataloader, criterion, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, criterion, optimizer
    )

    # Keep unwrapped model for easier access to individual components
    unwrapped_model = accelerator.unwrap_model(model)

    # Get number of training steps
    # Dataloader already per GPU so no need to divide by number of processes
    total_num_steps = cfg.optim.epochs * len(train_dataloader)
    warmup_num_steps = cfg.optim.warmup_epochs * len(train_dataloader)

    # Start training
    global_step: int = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        for batch in train_dataloader:

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
                    param_group["lr"] = lr * param_group["multiplier"]

                # Forward pass
                imgs = batch["image"]
                mask_logits_per_block, class_logits_per_block = model(imgs)

                # Loss calculation
                B = imgs.shape[0]
                targets = [{
                    "labels": torch.arange(1, len(labels) + 1).to(imgs.device),
                    "masks": torch.stack([batch[l][i].squeeze(0) for l in labels], dim=0).to(imgs.device),
                } for i in range(B)]  # remove channel dimension added by transforms
                losses_all_blocks = {}
                for i, (mask_logits, class_logits) in enumerate(
                    list(zip(mask_logits_per_block, class_logits_per_block))
                ):
                    losses = criterion(
                        masks_queries_logits=mask_logits, 
                        class_queries_logits=class_logits, 
                        targets=targets,
                    )
                    losses = {f"{k}_block_{i:02}": v for k, v in losses.items()}
                    losses_all_blocks |= losses
                final_loss = criterion.loss_total(losses_all_blocks)

                # Backward pass
                accelerator.backward(final_loss)

                # Update model
                if cfg.optim.clip_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unwrapped_model.parameters(),
                        cfg.optim.clip_grad_norm
                    )
                optimizer.step()

                # Log loss, lr, and weight decay
                if global_step % cfg.train.log_freq == 0:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{cfg.optim.epochs}, "
                        f"Step {global_step + 1}/{total_num_steps}, "
                        f"Loss: {final_loss.item():8f}, "
                        f"LR: {lr}, "
                    )
                    accelerator.log(
                        {
                            "loss": final_loss.item(),
                            "lr": lr,
                        },
                        step=global_step,
                    )
                
                # Zero gradients
                optimizer.zero_grad()

                # Update global step
                global_step += 1

        # Evaluate model
        model.eval()
        best_dice: float = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                # Forward pass
                imgs = batch["image"]
                mask_logits_per_block, class_logits_per_block = model(imgs)

                B = imgs.shape[0]
                targets = [{
                    "labels": torch.arange(1, len(labels) + 1).to(imgs.device),
                    "masks": torch.stack([batch[l][i].squeeze(0) for l in labels], dim=0).to(imgs.device),
                } for i in range(B)]  # remove channel dimension added by transforms
                targets = to_per_pixel_targets_semantic(targets, ignore_idx=0)

                for i, (mask_logits, class_logits) in enumerate(
                    list(zip(mask_logits_per_block, class_logits_per_block))
                ):
                    mask_logits = F.interpolate(
                        mask_logits,
                        size=imgs.shape[-3:],
                        mode="trilinear",
                    )
                    logits = to_per_pixel_logits_semantic(mask_logits, class_logits)

                    # Take the argmax to get class predictions
                    y_pred = torch.argmax(logits, dim=1)

                    # Compute Dice score
                    dice(y_pred=y_pred, y=targets[i])

        # Get predictions and labels form all devices
        val_dice = dice.aggregate().item()
        dice.reset()

        accelerator.print(f"Validation Dice: {val_dice:.4f}")
        accelerator.log({
            "val_dice": val_dice,
        }, step=global_step - 1)

        if val_dice > best_dice:
            best_dice = val_dice
            if accelerator.is_main_process:
                torch.save(
                    unwrapped_model.state_dict(), 
                    os.path.join(cfg.train.output_dir, f"best_model.pt")
                )
        accelerator.wait_for_everyone()

    # Make sure the trackers are finished before exiting
    accelerator.end_training()


def to_per_pixel_logits_semantic(
    mask_logits: torch.Tensor, class_logits: torch.Tensor
):
    return torch.einsum(
        "bqhwd, bqc -> bchwd",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )


def to_per_pixel_targets_semantic(
    targets: list[dict],
    ignore_idx,
):
    per_pixel_targets = []
    for target in targets:
        per_pixel_target = torch.full(
            target["masks"].shape[-3:],
            ignore_idx,
            dtype=target["labels"].dtype,
            device=target["labels"].device,
        )

        for i, mask in enumerate(target["masks"]):
            per_pixel_target[mask] = target["labels"][i]

        per_pixel_targets.append(per_pixel_target)

    return per_pixel_targets


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg, accelerator = setup(args, load_config("total_segmentator"))
    main(cfg, accelerator)
