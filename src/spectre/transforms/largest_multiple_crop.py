from typing import Sequence

import numpy as np
from monai.config import KeysCollection
from monai.transforms import Cropd, CenterSpatialCrop


class LargestMultipleCenterCropd(Cropd):
    """
    Dictionary-based transform for channel-first arrays only.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: sequence of ints, e.g. (128, 128, 64). Number of components must match image spatial dims.
        allow_missing_keys: don't raise if key missing.
        lazy: whether the internal cropper should be lazy.
    """
    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        self.patch_size = patch_size
        cropper = CenterSpatialCrop(roi_size=patch_size, lazy=lazy)  # Placeholder, will be reset per image
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
    
    def compute_largest_multiple_crop(self, img: np.ndarray) -> np.ndarray:
        
        spatial_dims = img.shape[1:]  # Exclude channel dim
        patch_size = np.asarray(self.patch_size, dtype=int)
        multiples = np.array(spatial_dims) // patch_size
        crop_size = (multiples * patch_size).astype(int)
        # If axis smaller than patch (multiple == 0) keep original axis size
        crop_size = np.where(crop_size == 0, spatial_dims, crop_size)
        return tuple(int(x) for x in crop_size)

    def __call__(self, data: dict, lazy: bool | None = None) -> dict:
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy

        for key in self.key_iterator(d):

            # reset cropper based on current image shape
            self.cropper.roi_size = self.compute_largest_multiple_crop(d[key].cpu().numpy())
            d[key] = self.cropper(d[key], lazy=lazy_)  # type: ignore
            
        return d
