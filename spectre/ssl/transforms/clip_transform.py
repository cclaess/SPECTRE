from typing import Tuple

from monai.transforms import Compose

from spectre.ssl.transforms import SWSpatialCropSamplesd


class CLIPTransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
        ):
        super().__init__(
            [
                # Crop the volume into equal non-overlapping samples
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                )
            ]
        )
