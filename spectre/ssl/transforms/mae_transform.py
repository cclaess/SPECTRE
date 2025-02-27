from typing import Tuple

from monai.transforms import (
    Compose,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    Resized,
)


class DINOTransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
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