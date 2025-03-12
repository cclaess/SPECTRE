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
    CopyItemsd,
    RandSpatialCropd,
    Resized,
)


class DINOTransform(Compose):
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 64),
        local_crop_size: Tuple[int, int, int] = (48, 48, 24),
        num_global_crops: int = 2,
        num_local_crops: int = 8,
    ):

        global_image_keys = ["image"] + [
            f"image_{i}" for i in range(2, num_global_crops + 1)
        ]
        if num_local_crops > 0:
            local_image_keys = ["image_local"] + [
                f"image_local_{i}" for i in range(2, num_local_crops + 1)
            ]
        else:
            local_image_keys = []

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
                    clip=True,
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
                CopyItemsd(
                    keys=("image",),
                    times=num_global_crops + num_local_crops - 1,
                    names=global_image_keys[1:] + local_image_keys,
                ),
                RandSpatialCropd(
                    keys=global_image_keys,
                    roi_size=tuple(int(sz * 0.4) for sz in input_size),
                    max_roi_size=input_size,
                    random_center=True,
                    random_size=True,
                ),
                RandSpatialCropd(
                    keys=local_image_keys,
                    roi_size=tuple(int(sz * 0.05) for sz in input_size),
                    max_roi_size=tuple(int(sz * 0.4) for sz in input_size),
                    random_center=True,
                    random_size=True,
                ),
                Resized(keys=global_image_keys, spatial_size=input_size),
                Resized(keys=local_image_keys, spatial_size=local_crop_size),
            ]
        )
