from copy import deepcopy
from typing import Tuple, Mapping, Hashable, Any, List

import torch
import numpy as np
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
    RandSpatialCropSamples,
    Resize,
    MapTransform,
    Randomizable,
    LazyTransform,
)


class DINOTransform(Compose):
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 64),
        local_crop_size: Tuple[int, int, int] = (48, 48, 24),
        num_global_crops: int = 2,
        num_local_crops: int = 8,
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
                    clip=True,
                ),
                Orientationd(keys=("image",), axcodes="RAS"),
                Spacingd(
                    keys=("image",), 
                    pixdim=(0.75, 0.75, 1.5), 
                    mode=("bilinear",),
                ),
                ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, -1)),
                SpatialPadd(keys=("image",), spatial_size=(-1, -1, input_size[2])),
                CastToTyped(keys=("image",), dtype=getattr(torch, dtype)),
                DINORandomCropTransformd(
                    keys=("image",),
                    input_size=input_size,
                    local_crop_size=local_crop_size,
                    num_base_patches=36,
                    num_global_crops=num_global_crops,
                    num_local_crops=num_local_crops,
                    dtype=dtype,
                ),
            ]
        )


class DINORandomCropTransformd(Randomizable, MapTransform, LazyTransform):
    """
    Custom transform for DINO-style cropping on 3D CT scans.
    
    The transform performs:
    
      1. A first random cropping to extract `num_base_patches` from the input image,
         each of size `input_size` (e.g. 128x128x64).
      2. For each base patch, it generates:
           - Global crops: using a random crop whose minimum size is 0.4 * input_size and maximum is input_size,
             and then resized back to input_size.
           - Local crops: using a random crop whose minimum size is 0.05 * input_size and maximum is 0.4 * input_size,
             and then resized to `local_crop_size`.
      3. The transform returns a list of dictionaries (one per base patch) where each dictionary
         contains:
             - "image": the base patch (cropped from the full scan)
             - "global_crops": list of the resized global crop tensors
             - "local_crops": list of the resized local crop tensors
         Other keys from the original dictionary are copied over.
    
    This implementation is structured similarly to MONAI's RandSpatialCropSamplesd.
    """
    
    def __init__(
        self,
        keys: Tuple[str, ...] = ("image",),
        input_size: Tuple[int, int, int] = (128, 128, 64),
        local_crop_size: Tuple[int, int, int] = (48, 48, 24),
        num_base_patches: int = 36,
        num_global_crops: int = 2,
        num_local_crops: int = 8,
        dtype: str = "float32",
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys)
        LazyTransform.__init__(self, lazy)
        self.input_size = input_size
        self.local_crop_size = local_crop_size
        self.num_base_patches = num_base_patches
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.resize_global = Resize(
            spatial_size=input_size, 
            mode="trilinear",
            dtype=getattr(torch, dtype),
        )
        self.resize_local = Resize(
            spatial_size=local_crop_size, 
            mode="trilinear",
            dtype=getattr(torch, dtype),
        )

    def randomize(self, data: Any = None) -> None:
        # Set a sub-seed for consistency across the different cropping operations.
        self.sub_seed = self.R.randint(0, 2**32 // 2 - 1)

    def __call__(self, data: Mapping[Hashable, Any], lazy: bool | None = None) -> List[dict]:
        lazy_ = self.lazy if lazy is None else lazy
        self.randomize(data)
        key = self.keys[0]
        image = data[key]

        # === Step 1: Generate base patches ===
        base_cropper = RandSpatialCropSamples(
            roi_size=self.input_size,
            num_samples=self.num_base_patches,
            random_center=True,
            random_size=False,
            lazy=lazy_,
        )
        base_cropper.set_random_state(seed=self.sub_seed)
        base_patches = list(base_cropper(image, lazy=lazy_))
        
        output = []
        # For each base patch, perform the global and local cropping.
        for patch in base_patches:
            # --- Global crops ---
            global_roi_size = tuple(int(s * 0.54) for s in self.input_size)  # 0.54 = (0.4 ** 2) ** (1/3)
            global_cropper = RandSpatialCropSamples(
                roi_size=global_roi_size,
                num_samples=self.num_global_crops,
                max_roi_size=self.input_size,
                random_center=True,
                random_size=True,
                lazy=lazy_,
            )
            global_cropper.set_random_state(seed=self.sub_seed)
            global_crops = list(global_cropper(patch, lazy=lazy_))
            # Resize each global crop back to the original input size.
            resized_global = [
                self.resize_global(gc)
                for gc in global_crops
            ]
            
            # --- Local crops ---
            local_roi_size = tuple(int(s * 0.14) for s in self.input_size)  # 0.14 = (0.05 ** 2) ** (1/3)
            max_local_roi = tuple(int(s * 0.54) for s in self.input_size)  # 0.54 = (0.4 ** 2) ** (1/3)
            local_cropper = RandSpatialCropSamples(
                roi_size=local_roi_size,
                num_samples=self.num_local_crops,
                max_roi_size=max_local_roi,
                random_center=True,
                random_size=True,
                lazy=lazy_,
            )
            local_cropper.set_random_state(seed=self.sub_seed)
            local_crops = list(local_cropper(patch, lazy=lazy_))
            # Resize each local crop to the desired local crop size.
            resized_local = [
                self.resize_local(lc)
                for lc in local_crops
            ]
            
            # Create the output dictionary for this base patch.
            # We deepcopy the original dictionary to include any other keys unchanged.
            patch_dict = deepcopy(data)
            patch_dict[key] = patch  # assign the base patch to the main key
            patch_dict["global_crops"] = resized_global
            patch_dict["local_crops"] = resized_local
            output.append(patch_dict)
            
        return output


if __name__ == "__main__":

    # Save some example data after transforming it.
    import os
    import SimpleITK as sitk

    data = {"image": r"data/test_data/train_1_a_1.nii.gz"}
    transform = DINOTransform()
    transformed_data = transform(data)

    # Save the different crops to a folder for visualization.
    output_dir = r"data/test_data/dino_transform_output"
    os.makedirs(output_dir, exist_ok=True)

    for i, patch in enumerate(transformed_data):

        # Save the global crops
        for j, gc in enumerate(patch["global_crops"]):
            gc_img = sitk.GetImageFromArray(gc.squeeze(0).numpy())
            gc_img.SetSpacing((1.5, 0.75, 0.75))
            gc_path = os.path.join(output_dir, f"{i}_global_crop_{j}.nii.gz")
            sitk.WriteImage(gc_img, gc_path)

        # Save the local crops
        for j, lc in enumerate(patch["local_crops"]):
            lc_img = sitk.GetImageFromArray(lc.squeeze(0).numpy())
            lc_img.SetSpacing((1.5, 0.75, 0.75))
            lc_path = os.path.join(output_dir, f"{i}_local_crop_{j}.nii.gz")
            sitk.WriteImage(lc_img, lc_path)
