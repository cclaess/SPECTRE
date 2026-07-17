"""Pure-torch CT preprocessing: HU scaling, centre cropping and grid patching.

This module is vendored into the Hugging Face Hub repository by `scripts/export_hf.py`,
so it must import nothing beyond `torch` and the standard library.

The functions here reproduce the MONAI pipeline used for pretraining and evaluation
(`ScaleIntensityRanged` -> `LargestMultipleCenterCropd` -> `GridPatchd`) exactly. Two details
are load-bearing and are covered by `tests/test_windowing_parity.py`:

* Patch order is C-order with depth varying fastest, i.e. crop `n` is at grid position
  `n = (h * n_w + w) * n_d + d`. This matches both MONAI's `GridPatch` and the coordinates
  `RotaryPositionEmbedding` builds, so getting it wrong shifts every embedding silently.
* The centre-crop offset is `start = (S // 2) - (R // 2)`, which is not the same as
  `(S - R) // 2` when the crop is odd and the volume even.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

DEFAULT_CROP_SIZE: Tuple[int, int, int] = (128, 128, 64)  # (H, W, D)
DEFAULT_HU_RANGE: Tuple[float, float] = (-1000.0, 1000.0)

__all__ = [
    'DEFAULT_CROP_SIZE',
    'DEFAULT_HU_RANGE',
    'scale_intensity_range',
    'largest_multiple_crop_size',
    'largest_multiple_center_crop',
    'grid_patch',
    'window_scan',
]


def scale_intensity_range(
    x: torch.Tensor,
    a_min: float = DEFAULT_HU_RANGE[0],
    a_max: float = DEFAULT_HU_RANGE[1],
    b_min: float = 0.0,
    b_max: float = 1.0,
    clip: bool = True,
) -> torch.Tensor:
    """Map intensities from `[a_min, a_max]` onto `[b_min, b_max]`.

    Mirrors `monai.transforms.ScaleIntensityRange`. SPECTRE was pretrained with the default
    HU window of [-1000, 1000] -> [0, 1] with clipping.
    """
    if a_max == a_min:
        raise ValueError(f"a_min and a_max must differ, got a_min == a_max == {a_min}.")
    x = (x - a_min) / (a_max - a_min)
    x = x * (b_max - b_min) + b_min
    if clip:
        x = torch.clamp(x, min(b_min, b_max), max(b_min, b_max))
    return x


def largest_multiple_crop_size(
    spatial_shape: Sequence[int],
    crop_size: Sequence[int] = DEFAULT_CROP_SIZE,
) -> Tuple[int, ...]:
    """Largest per-axis size that is a whole multiple of `crop_size`.

    An axis shorter than its crop keeps its original size (there is no non-zero multiple to
    take), matching `spectre.transforms.LargestMultipleCenterCropd`.
    """
    if len(spatial_shape) != len(crop_size):
        raise ValueError(
            f"spatial_shape {tuple(spatial_shape)} and crop_size {tuple(crop_size)} must have "
            f"the same number of dimensions."
        )
    return tuple(
        (s // c) * c if s >= c else s
        for s, c in zip(spatial_shape, crop_size)
    )


def largest_multiple_center_crop(
    x: torch.Tensor,
    crop_size: Sequence[int] = DEFAULT_CROP_SIZE,
) -> torch.Tensor:
    """Centre-crop a channel-first `(C, H, W, D)` volume to a whole multiple of `crop_size`."""
    if x.ndim != 4:
        raise ValueError(f"Expected a channel-first (C, H, W, D) volume, got shape {tuple(x.shape)}.")

    spatial_shape = tuple(x.shape[1:])
    roi_size = largest_multiple_crop_size(spatial_shape, crop_size)

    # MONAI's CenterSpatialCrop offset: roi_center - (roi_size // 2), clamped at 0. This is not
    # the same as (S - R) // 2 for odd R and even S, so keep the formula verbatim.
    slices = []
    for size, roi in zip(spatial_shape, roi_size):
        start = max(size // 2 - roi // 2, 0)
        slices.append(slice(start, start + roi))
    return x[(slice(None), *slices)]


def grid_patch(
    x: torch.Tensor,
    crop_size: Sequence[int] = DEFAULT_CROP_SIZE,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Tile a `(C, H, W, D)` volume into non-overlapping crops.

    Every spatial axis must already be a whole multiple of `crop_size` (see
    `largest_multiple_center_crop`).

    Returns:
        The crops `(N, C, cH, cW, cD)` and the grid `(n_h, n_w, n_d)` with `N == n_h*n_w*n_d`.
        Crop `n` sits at grid position `n = (h * n_w + w) * n_d + d`.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a channel-first (C, H, W, D) volume, got shape {tuple(x.shape)}.")

    channels = x.shape[0]
    spatial_shape = tuple(x.shape[1:])
    c_h, c_w, c_d = (int(c) for c in crop_size)

    for axis, (size, crop) in enumerate(zip(spatial_shape, (c_h, c_w, c_d))):
        if crop <= 0:
            raise ValueError(f"crop_size must be positive, got {tuple(crop_size)}.")
        if size % crop != 0:
            raise ValueError(
                f"Axis {'HWD'[axis]} has size {size}, which is not a whole multiple of the crop "
                f"size {crop}. Call largest_multiple_center_crop() first, or use window_scan() "
                f"which does this for you."
            )

    n_h, n_w, n_d = (s // c for s, c in zip(spatial_shape, (c_h, c_w, c_d)))

    # Depth varies fastest, matching MONAI's GridPatch and the RoPE coordinate order.
    crops = x.contiguous().view(channels, n_h, c_h, n_w, c_w, n_d, c_d)
    crops = crops.permute(1, 3, 5, 0, 2, 4, 6)
    crops = crops.reshape(n_h * n_w * n_d, channels, c_h, c_w, c_d)
    return crops, (n_h, n_w, n_d)


def _pad_short_axes(
    x: torch.Tensor,
    crop_size: Sequence[int],
    pad_value: float,
) -> torch.Tensor:
    """Symmetrically pad any spatial axis shorter than its crop up to exactly one crop."""
    spatial_shape = tuple(x.shape[1:])
    short = [
        (axis, size, int(crop))
        for axis, (size, crop) in enumerate(zip(spatial_shape, crop_size))
        if size < crop
    ]
    if not short:
        return x

    # F.pad takes the last dimension first, so build the spec back to front: D, W, H.
    pad_spec: list = []
    for axis in reversed(range(3)):
        size, crop = spatial_shape[axis], int(crop_size[axis])
        if size < crop:
            total = crop - size
            pad_spec.extend([total // 2, total - total // 2])
        else:
            pad_spec.extend([0, 0])

    detail = ', '.join(
        f"{'HWD'[axis]}={size} < {crop}" for axis, size, crop in short
    )
    warnings.warn(
        f"Scan is smaller than one crop along {detail}. Padding with {pad_value} to reach the "
        f"crop size {tuple(int(c) for c in crop_size)}. SPECTRE was not pretrained on padded "
        f"volumes, so the resulting embedding is out of distribution. Pass pad_short_axes=False "
        f"to raise instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return F.pad(x, pad_spec, mode='constant', value=pad_value)


def window_scan(
    x: torch.Tensor,
    crop_size: Sequence[int] = DEFAULT_CROP_SIZE,
    *,
    scale_intensity: bool = True,
    hu_range: Tuple[float, float] = DEFAULT_HU_RANGE,
    pad_short_axes: bool = True,
    pad_value: Optional[float] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Turn a whole CT volume into the crops SPECTRE's backbone consumes.

    Applies, in the order used for pretraining: HU scaling -> centre crop to a whole multiple of
    `crop_size` -> grid patching.

    Args:
        x: A `(H, W, D)` or `(C, H, W, D)` volume. In raw Hounsfield Units unless
            `scale_intensity=False`.
        crop_size: Spatial size `(H, W, D)` of one crop.
        scale_intensity: Map `hu_range` onto [0, 1] with clipping. Turn off if `x` is already
            normalised.
        hu_range: The `(a_min, a_max)` HU window. SPECTRE was pretrained with (-1000, 1000).
        pad_short_axes: Pad axes shorter than one crop instead of raising.
        pad_value: Value to pad with. Defaults to the scaled air value (0.0) when
            `scale_intensity` is on, and to `x.min()` otherwise.

    Returns:
        The crops `(N, C, cH, cW, cD)` and the grid `(n_h, n_w, n_d)`.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim != 4:
        raise ValueError(
            f"Expected a (H, W, D) or (C, H, W, D) volume, got shape {tuple(x.shape)}."
        )

    if len(crop_size) != 3:
        raise ValueError(f"crop_size must have 3 elements (H, W, D), got {tuple(crop_size)}.")

    if scale_intensity:
        if float(x.min()) > -100.0:
            warnings.warn(
                f"Input does not look like Hounsfield Units (min={float(x.min()):.1f}). It will "
                f"be rescaled from {hu_range} to [0, 1], which double-scales already-normalised "
                f"data. Pass scale_intensity=False if the volume is already normalised.",
                RuntimeWarning,
                stacklevel=2,
            )
        x = scale_intensity_range(x, a_min=hu_range[0], a_max=hu_range[1])

    if pad_value is None:
        pad_value = 0.0 if scale_intensity else float(x.min())

    too_short = [
        f"{'HWD'[axis]}={int(size)} < {int(crop)}"
        for axis, (size, crop) in enumerate(zip(x.shape[1:], crop_size))
        if size < crop
    ]
    if too_short:
        if not pad_short_axes:
            raise ValueError(
                f"Scan is smaller than one crop along {', '.join(too_short)}. Pass "
                f"pad_short_axes=True to pad with air, or supply a scan covering at least "
                f"{tuple(int(c) for c in crop_size)} voxels."
            )
        x = _pad_short_axes(x, crop_size, pad_value)

    x = largest_multiple_center_crop(x, crop_size)
    return grid_patch(x, crop_size)
