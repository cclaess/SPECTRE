from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    Resized,
)


class MAETransform(Compose):
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

                # Take equal amount of crops from the same subject as could 
                # be taken without overlap to avoid data-loading overhead
                RandSpatialCropSamplesd(
                    keys=("image",),
                    roi_size=input_size,
                    num_samples=36,
                    random_center=True,
                    random_size=False,
                ),
                # Do a random resized crop
                RandSpatialCropd(
                    keys=("image",),
                    roi_size=tuple(int(sz * 0.2) for sz in input_size),
                    max_roi_size=input_size,
                    num_samples=2,
                    random_center=True,
                    random_size=True,
                ),
                Resized(keys=("image",), spatial_size=input_size),
            ]
        )
