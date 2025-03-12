from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
)

from spectre.transforms import SWSpatialCropSamplesd


class CLIPTransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
        ):
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
                ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, 256)),
                
                # Crop the volume into equal non-overlapping samples
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                )
            ]
        )
