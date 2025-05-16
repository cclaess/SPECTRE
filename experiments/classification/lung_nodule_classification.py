import os
import math
import argparse
from pathlib import Path
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from accelerate import Accelerator, DataLoaderConfiguration
from torch.optim import AdamW
from torch.utils.data import Dataset, Subset
from monai.data import MetaTensor, DataLoader
from monai.metrics import compute_roc_auc
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Spacing,
    RandAxisFlip,
    RandRotate,
    ResizeWithPadOrCrop,
)

import spectre.models as models
from spectre.utils.config import setup
from spectre.configs import load_config
from spectre.utils.scheduler import cosine_warmup_schedule


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Pretrain DINO")
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

    # Get dataloader
    dataset = LUNA25Dataset(
        cfg.train.dataset_path,
        pd.read_csv(cfg.train.dataset),
        transform=LUNA25PatchTransform(),
    )
    train_dataset = Subset(dataset, range(1070, len(dataset)))
    val_dataset = Subset(dataset, range(1070))

    num_pos_samples = sum(dataset.labels)
    num_neg_samples = len(dataset.labels) - num_pos_samples
    pos_weight = torch.tensor([num_neg_samples / num_pos_samples])

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
    
    # Initialize model
    if (
        cfg.model.architecture in models.__dict__ 
        and cfg.model.architecture.startswith("vit")
    ):
        model = models.__dict__[cfg.model.architecture](
            num_classes=1,
        )
    elif (
        cfg.model.architecture in models.__dict__
        and cfg.model.architecture.startswith("resnet")
        or cfg.model.architecture.startswith("resnext")
    ):
        model = models.__dict__[cfg.model.architecture](
            num_classes=1,
            norm_layer=partial(nn.BatchNorm3d, track_running_stats=False),
        )
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")
    
    # Load pretrained weights and freeze backbone if specified
    if cfg.model.pretrained_weights is not None:
        msg = model.load_state_dict(torch.load(cfg.model.pretrained_weights), strict=False)
        accelerator.print(f"Pretrained weights loaded with message: {msg}")
        if cfg.model.linear_only:
            for name, param in model.named_parameters():
                if name not in ["fc.weight", "fc.bias", "head.weight", "head.bias"]:
                    param.requires_grad = False

    # Initialize criterion
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )

    # Prepare model, data, and optimizer for training
    model, train_dataloader, val_dataloader, criterion, optimizer = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
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
                    param_group["lr"] = lr

                # Forward pass
                output = model(batch["image"])
                loss = criterion(output, batch["label"])

                # Backward pass
                accelerator.backward(loss)

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
                        f"Loss: {loss.item():8f}, "
                        f"LR: {lr}, "
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

        # Evaluate model
        model.eval()
        predictions = torch.tensor([], device=accelerator.device)
        labels = torch.tensor([], device=accelerator.device)
        best_auc: float = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                output = model(batch["image"])

                # Keep all outputs for later analysis
                predictions = torch.cat((predictions, output.detach()), dim=0)
                labels = torch.cat((labels, batch["label"]), dim=0)

        # Get predictions and labels form all devices
        predictions = accelerator.gather(predictions)
        labels = accelerator.gather(labels)
            
        val_loss = criterion(predictions, labels)
        val_loss = val_loss.item()

        val_auc = compute_roc_auc(predictions.cpu(), labels.cpu())
            
        accelerator.print(f"Validation loss: {val_loss:.4f}")
        accelerator.print(f"Validation AUC: {val_auc:.4f}")
        accelerator.log({
            "val_loss": val_loss,
            "val_auc": val_auc,
        }, step=global_step - 1)

        if val_auc > best_auc:
            best_auc = val_auc
            if accelerator.is_main_process:
                torch.save(
                    unwrapped_model.state_dict(), 
                    os.path.join(cfg.train.output_dir, f"best_model.pt")
                )
        accelerator.wait_for_everyone()

    # Make sure the trackers are finished before exiting
    accelerator.end_training()


class LUNA25Dataset(Dataset):
    """LUNA25 dataset
            Args:
            data_dir (str): path to the nodule_blocks data directory
            dataset (pd.DataFrame): dataframe with the dataset information
            translations (bool): whether to apply random translations
            rotations (tuple): tuple with the rotation ranges
            size_px (int): size of the patch in pixels
            size_mm (int): size of the patch in mm
            mode (str): 2D or 3D

    """

    def __init__(
        self,
        data_dir: str,
        dataset: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):

        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.transform = transform

        # return a list of the classes per sample
        self.labels = self.dataset["label"].tolist()

    def __getitem__(self, idx):  # caseid, z, y, x, label, radius

        pd = self.dataset.iloc[idx]

        label = pd.label

        annotation_id = pd.AnnotationID

        image_path = self.data_dir / "image" / f"{annotation_id}.npy"
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        # numpy memory map data/image case file
        img = np.load(image_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()

        origin = metadata["origin"]
        spacing = metadata["spacing"]
        # transform = metadata["transform"]

        # change orientation to RAS
        img = np.flip(np.transpose(img, (2, 1, 0)), axis=(0, 1))  # (S, P, L) -> (L, P, S) -> (R, A, S)
        spacing = spacing[::-1]
        origin = origin[::-1]
        origin[:2] *= -1

        # to MetaTensor
        img = torch.from_numpy(img.copy()).float()
        affine = torch.eye(4)
        affine[0, 0], affine[1, 1], affine[2, 2] = spacing[0], spacing[1], spacing[2]
        affine[:3, 3] = torch.from_numpy(origin.copy())
        img = MetaTensor(img, affine=affine)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": torch.ones((1,)) * label,
            "ID": annotation_id,
        }

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str
    

class LUNA25PatchTransform(Compose):
    def __init__(self):
        super().__init__([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            Spacing(pixdim=(0.75, 0.75, 1.5), mode="trilinear", lazy=True),
            RandRotate(range_x=math.pi / 6, range_y=math.pi / 6, range_z=math.pi / 6, prob=0.5, lazy=True),
            # RandAxisFlip(prob=0.5, lazy=True),
            ResizeWithPadOrCrop((128, 128, 64)),
        ])


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg, accelerator = setup(args, load_config("classification_test"))
    main(cfg, accelerator)
