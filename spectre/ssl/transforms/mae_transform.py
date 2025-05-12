from typing import Tuple

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    SpatialPadd,
    CastToTyped,
    ResizeWithPadOrCropd,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    Resized,
)


class MAETransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
            dtype: str = "float32",
        ):
        assert dtype in ["float16", "float32"], "dtype must be either 'float16' or 'float32'"
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
                ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, -1)),
                SpatialPadd(keys=("image",), spatial_size=(-1, -1, input_size[2])),
                CastToTyped(keys=("image",), dtype=getattr(torch, dtype)),
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
                    roi_size=tuple(int(sz * 0.34) for sz in input_size),  # 0.34 = (0.2 ** 2) ** (1/3)
                    max_roi_size=input_size,
                    random_center=True,
                    random_size=True,
                ),
                Resized(keys=("image",), spatial_size=input_size),
            ]
        )


if __name__ == "__main__":

    # Save some example data after transforming it.
    import os
    import SimpleITK as sitk

    data = {"image": r"data/test_data/train_1_a_1.nii.gz"}
    transform = MAETransform()
    transformed_data = transform(data)

    # Save the different crops to a folder for visualization.
    output_dir = r"data/test_data/mae_transform_output"
    os.makedirs(output_dir, exist_ok=True)

    for i, patch in enumerate(transformed_data):

        # Save the crops
        patch_img = sitk.GetImageFromArray(patch["image"].squeeze(0).numpy())
        patch_img.SetSpacing((1.5, 0.75, 0.75))
        patch_path = os.path.join(output_dir, f"{i}_crop.nii.gz")
        sitk.WriteImage(patch_img, patch_path)
