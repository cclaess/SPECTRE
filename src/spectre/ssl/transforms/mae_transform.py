from typing import Tuple

import torch

MONAI_IMPORT_ERROR = None
try:
    import monai.transforms as transforms
except ImportError as e:
    transforms = None  # type: ignore
    MONAI_IMPORT_ERROR = e


if transforms is not None:
    Compose = transforms.Compose
else:
    Compose = object  # type: ignore


class MAETransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
            dtype: str = "float32",
        ):
        if MONAI_IMPORT_ERROR is not None:
            raise ImportError(
                "MONAI is required to use MAETransform but not installed. "
                "Please install MONAI to use this transform."
            ) from MONAI_IMPORT_ERROR
        
        assert dtype in ["float16", "float32"], "dtype must be either 'float16' or 'float32'"
        super().__init__(
            [
                transforms.LoadImaged(keys=("image",)),
                transforms.EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
                transforms.ScaleIntensityRanged(
                    keys=("image",), 
                    a_min=-1000, 
                    a_max=1000, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True
                ),
                transforms.Orientationd(keys=("image",), axcodes="RAS"),
                transforms.Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
                transforms.ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, -1)),
                transforms.SpatialPadd(keys=("image",), spatial_size=(-1, -1, input_size[2])),
                transforms.CastToTyped(keys=("image",), dtype=getattr(torch, dtype)),
                transforms.RandSpatialCropSamplesd(
                    keys=("image",),
                    roi_size=input_size,
                    num_samples=36,
                    random_center=True,
                    random_size=False,
                ),
                # Do a random resized crop
                transforms.RandSpatialCropd(
                    keys=("image",),
                    roi_size=tuple(int(sz * 0.34) for sz in input_size),  # 0.34 = (0.2 ** 2) ** (1/3)
                    max_roi_size=input_size,
                    random_center=True,
                    random_size=True,
                ),
                transforms.Resized(keys=("image",), spatial_size=input_size),
            ]
        )
