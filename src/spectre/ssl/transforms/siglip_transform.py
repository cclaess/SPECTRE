from typing import Tuple

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    EnsureTyped,
    RandSpatialCropd,
    RandFlipd,
    GridPatchd,
    SelectItemsd,
)

from spectre.transforms import RandomReportTransformd


class SigLIPTransform(Compose):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (384, 384, 256),
        image_pixdim: Tuple[float, float, float] = (0.75, 0.75, 1.5),
        sliding_window_size: Tuple[int, int, int] = (128, 128, 64),
        max_shift: Tuple[int, int, int] = (64, 64, 32),
        max_num_icd10: int = 20,
        keep_original_prob: float = 0.5,
        drop_prob: float = 0.3,
        dtype: str = "float32",
        use_gds: bool = False,
    ):

        assert dtype in ["float16", "float32"], \
            "dtype must be either 'float16' or 'float32'"
        
        device = "cuda" if (use_gds and torch.cuda.is_available()) else "cpu"
        base_crop_size = tuple(
            image_size[i] + 2 * max_shift[i] for i in range(3)
        )

        super().__init__([
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(
                keys=("image",), 
                channel_dim="no_channel"
            ),
            ScaleIntensityRanged(
                keys=("image",), 
                a_min=-1000, 
                a_max=1000, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True
            ),
            Orientationd(keys=("image",), axcodes="RAS"),
            Spacingd(
                keys=("image",), 
                pixdim=image_pixdim, 
                mode=("bilinear",)
            ),
            ResizeWithPadOrCropd(
                keys=("image",), 
                spatial_size=base_crop_size
            ),
            EnsureTyped(
                keys=("image",), 
                dtype=getattr(torch, dtype), 
                device=device
            ),
            RandSpatialCropd(
                keys=("image",),
                roi_size=image_size,
                random_size=False,
            ),
            RandFlipd(keys=("image",), spatial_axis=0, prob=0.5),
            RandFlipd(keys=("image",), spatial_axis=1, prob=0.5),
            RandFlipd(keys=("image",), spatial_axis=2, prob=0.5),
            GridPatchd(
                keys=("image",), 
                patch_size=sliding_window_size, 
                overlap=0.0, 
            ),
            RandomReportTransformd(
                keys=("findings", "impressions", "icd10"), 
                max_num_icd10=max_num_icd10, 
                keep_original_prob=keep_original_prob,
                drop_prob=drop_prob,
            ),
            SelectItemsd(
                keys=("image", "report"),
            ),
        ])
