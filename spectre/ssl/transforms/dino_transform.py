from typing import Tuple

from monai.transforms import (
    Compose,
    RandSpatialCropSamplesd,
    CopyItemsd,
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
        super().__init__(
            [
                # Take equal amount of crops from the same subject as could 
                # be taken without overlap to avoid data-loading overhead
                RandSpatialCropSamplesd(
                    keys=("image",),
                    roi_size=input_size,
                    num_samples=36,
                    random_center=True,
                    random_size=False,
                ),
                # Make two copies, one for global and one for local views
                CopyItemsd(keys=("image",), times=1, names=("image_local",)),
                # Now make two global random crops
                RandSpatialCropSamplesd(
                    keys=("image",),
                    roi_size=tuple(int(sz * 0.4) for sz in input_size),
                    max_roi_size=input_size,
                    num_samples=num_global_crops,
                    random_center=True,
                    random_size=True,
                ),
                Resized(keys=("image",), spatial_size=input_size),
                # Now make eight local random crops
                RandSpatialCropSamplesd(
                    keys=("image_local",),
                    roi_size=tuple(int(sz * 0.05) for sz in input_size),
                    max_roi_size=tuple(int(sz * 0.4) for sz in input_size),
                    num_samples=num_local_crops,
                    random_center=True,
                    random_size=True,
                ),
                Resized(keys=("image_local",), spatial_size=local_crop_size),
            ]
        )
