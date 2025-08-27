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
    DeleteItemsd,
)

from spectre.transforms import GenerateReportTransform


class SigLIPTransform(Compose):
    def __init__(
        self, 
        input_size: Tuple[int, int, int] = (128, 128, 64),
        dtype: str = "float32",
        use_gds: bool = False,
    ):
        global_size = (
            384 + input_size[0],
            384 + input_size[1],
            256 + input_size[2],
        )

        assert dtype in ["float16", "float32"], \
            "dtype must be either 'float16' or 'float32'"
        
        device = "cuda" if (use_gds and torch.cuda.is_available()) else "cpu"

        super().__init__([
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=("image",), 
                a_min=-1000, 
                a_max=1000, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True
            ),
            Orientationd(keys=("image",), axcodes="RAS"),
            Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
            ResizeWithPadOrCropd(keys=("image",), spatial_size=global_size),
            EnsureTyped(keys=("image",), dtype=getattr(torch, dtype), device=device),
            RandSpatialCropd(
                keys=("image",),
                roi_size=(384, 384, 256),
                random_size=False,
            ),

            # load the text data
            GenerateReportTransform(
                keys=("findings", "impressions", "icd10"), 
                max_num_icd10=20, 
                likelihood_original=0.5,
                drop_chance=0.3,
            ),
            # Delete findings, impressions and icd10 to avoid errors with collate_fn
            DeleteItemsd(keys=("findings", "impressions", "icd10")),
        ])
