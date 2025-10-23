from typing import Sequence

import numpy as np
from monai.config import KeysCollection
from monai.transforms import Cropd, CenterSpatialCrop


class _DynamicCenterSpatialCrop(CenterSpatialCrop):
    """
    Runtime cropper for channel-first arrays only.

    Expects input arrays of shape (C, H, W, ...) where the number of spatial dims
    equals len(patch_size). Computes the largest centered crop whose spatial sizes are
    multiples of `patch_size` and <= original spatial size (per axis). If an axis is
    smaller than the patch dimension (multiple == 0) that axis is left unchanged.
    """
    def __init__(self, patch_size: Sequence[int], lazy: bool = False) -> None:
        super().__init__(roi_size=tuple(np.asarray(patch_size, int)), lazy=lazy)
        self.patch_size = np.asarray(patch_size, dtype=int)
        if self.patch_size.ndim != 1:
            raise ValueError("patch_size must be a 1D sequence like (128,128,64)")

    def __call__(self, img, lazy: bool | None = None):
        lazy_ = self.lazy if lazy is None else lazy

        arr = np.asarray(img)
        n_spatial = self.patch_size.size

        # Enforce channel-first only
        if arr.ndim != n_spatial + 1:
            raise ValueError(
                f"Expected channel-first array with ndim == {n_spatial + 1} "
                f"(C + {n_spatial} spatial dims). Got ndim {arr.ndim}."
            )

        spatial = np.array(arr.shape[1:], dtype=int)

        multiples = spatial // self.patch_size
        crop_size = (multiples * self.patch_size).astype(int)

        # If axis smaller than patch (multiple == 0) keep original axis size
        crop_size = np.where(crop_size == 0, spatial, crop_size)
        crop_size = tuple(int(x) for x in crop_size)

        temp_cropper = CenterSpatialCrop(roi_size=crop_size, lazy=lazy_)
        return temp_cropper(arr)


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
        cropper = _DynamicCenterSpatialCrop(patch_size=patch_size, lazy=lazy)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
