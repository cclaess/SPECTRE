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
    CastToTyped,
    RandSpatialCropd,
    RandGridPatchd,
)

from spectre.transforms import SWSpatialCropSamplesd, GenerateReportTransform


class SigLIPTransform(Compose):
    def __init__(
        self, 
        input_size: Tuple[int, int, int] = (128, 128, 64),
        dtype: str = "float32",
    ):
        assert dtype in ["float16", "float32"], "dtype must be either 'float16' or 'float32'"
        global_size = (
            384 + input_size[0] - 2,  # Avoid fitting a full patch by subtracting 2
            384 + input_size[1] - 2,
            256 + input_size[2] - 2,
        )
        super().__init__(
            [
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
                CastToTyped(keys=("image",), dtype=getattr(torch, dtype)),
                
                # Crop the volume into equal non-overlapping samples
                # RandSpatialCropd(
                #     keys=("image",),
                #     roi_size=(384, 384, 256),
                #     random_size=False,
                #     random_center=True,
                # ),
                # SWSpatialCropSamplesd(
                #     keys=("image",),
                #     patch_size=input_size,
                #     overlap=0.0,
                # ),
                RandGridPatchd(
                    keys=("image",),
                    patch_size=input_size,
                    min_offset=(0, 0, 0),
                    max_offset=tuple(sz - 2 for sz in input_size),  # Also subtract 2 here
                    num_patches=36,
                ),

                # load the text data
                GenerateReportTransform(
                    keys=("findings", "impressions", "icd10"), 
                    max_num_icd10=20, 
                    likelihood_original=0.5,
                    drop_chance=0.3,
                ),
            ]
        )
