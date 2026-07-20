"""Reading CT scans off disk.

Everything here needs optional dependencies, installed with:

    pip install "spectre-fm[inference]"

Importing this module always works; the error is raised when a function is actually called, so
`import spectre` keeps working on a base install. This module is deliberately *not* vendored into
the Hugging Face Hub repo (see `scripts/export_hf.py`), which is why `spectre.model` never
imports it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

from spectre.windowing import (
    DEFAULT_CROP_SIZE,
    DEFAULT_HU_RANGE,
    scale_intensity_range,
    window_scan,
)

NIBABEL_IMPORT_ERROR: Optional[BaseException] = None
try:
    import nibabel as nib
    import numpy as np
except ImportError as e:  # pragma: no cover - exercised only on a base install
    nib = None  # type: ignore
    np = None  # type: ignore
    NIBABEL_IMPORT_ERROR = e

__all__ = [
    'CTMeta',
    'find_ct_files',
    'load_ct',
    'resample',
    'load_and_window',
]

_INSTALL_HINT = (
    'Install them with:\n'
    '    pip install "spectre-fm[inference]"'
)


def _require_nibabel() -> None:
    if NIBABEL_IMPORT_ERROR is not None:
        raise ImportError(
            f"Reading NIfTI files requires optional dependencies (nibabel, numpy).\n{_INSTALL_HINT}"
        ) from NIBABEL_IMPORT_ERROR


@dataclass
class CTMeta:
    """Provenance for a loaded scan."""
    path: str
    affine: Any
    spacing: Tuple[float, float, float]
    orientation: str
    original_shape: Tuple[int, ...]


def find_ct_files(
    root: Union[str, os.PathLike],
    patterns: Sequence[str] = ("*.nii", "*.nii.gz"),
    recursive: bool = True,
) -> List[Path]:
    """Find NIfTI files under `root`, or return `[root]` if it is itself a file.

    Needs no optional dependencies.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"No such file or directory: {root}")

    found: List[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    # *.nii.gz also matches *.nii's glob on some platforms; de-duplicate and stabilise the order.
    return sorted(set(found))


def load_ct(
    path: Union[str, os.PathLike],
    *,
    orientation: Optional[str] = "RAS",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, CTMeta]:
    """Load a NIfTI scan as a channel-first `(1, H, W, D)` tensor of raw Hounsfield Units.

    Args:
        path: A .nii or .nii.gz file.
        orientation: Reorient to this axis code. Only "RAS" (what SPECTRE was pretrained on) and
            None are supported; reorientation is pure flips and transposes, so it changes no
            voxel values.
        dtype: Dtype of the returned tensor.

    Returns:
        The volume and its metadata.
    """
    _require_nibabel()

    if orientation not in ("RAS", None):
        raise ValueError(
            f"Only orientation='RAS' or None is supported, got {orientation!r}. SPECTRE was "
            f"pretrained on RAS."
        )

    image = nib.load(os.fspath(path))
    original_shape = tuple(int(s) for s in image.shape)

    if orientation == "RAS":
        image = nib.as_closest_canonical(image)

    # get_fdata (not dataobj) applies scl_slope/scl_inter, which is what makes the values HU.
    # float32 rather than the float64 default, which would double peak memory on a large scan.
    array = image.get_fdata(dtype=np.float32)
    # Reorienting a non-RAS scan flips axes, leaving a view with negative strides that
    # torch.from_numpy cannot take; ascontiguousarray both fixes that and is a no-op when the
    # scan was already RAS.
    volume = torch.from_numpy(np.ascontiguousarray(array)).to(dtype).unsqueeze(0)

    zooms = image.header.get_zooms()[:3]
    meta = CTMeta(
        path=os.fspath(path),
        affine=image.affine,
        spacing=tuple(float(z) for z in zooms),
        orientation="".join(nib.aff2axcodes(image.affine)),
        original_shape=original_shape,
    )
    return volume, meta


def resample(
    x: torch.Tensor,
    meta: CTMeta,
    spacing: Sequence[float],
    *,
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, CTMeta]:
    """Resample a `(C, H, W, D)` volume to a target voxel spacing in mm.

    Uses MONAI's `Spacing` so results match the eval scripts. MONAI is imported here rather than
    at module scope so a missing MONAI cannot break the default (no-resampling) path.
    """
    _require_nibabel()
    try:
        from monai.data import MetaTensor
        from monai.transforms import Spacing
    except ImportError as e:
        raise ImportError(
            f"Resampling to a target voxel spacing requires MONAI.\n{_INSTALL_HINT}"
        ) from e

    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 elements, got {tuple(spacing)}.")

    # Spacing reads the source spacing from the input's affine, so the affine has to travel on the
    # tensor as a MetaTensor - it is not a call argument (older MONAI took an `affine=` kwarg).
    volume = MetaTensor(x, affine=torch.as_tensor(meta.affine, dtype=torch.float64))
    spacer = Spacing(pixdim=tuple(float(s) for s in spacing), mode=mode)
    resampled = spacer(volume)
    affine = getattr(resampled, "affine", meta.affine)
    resampled = resampled.as_tensor() if hasattr(resampled, "as_tensor") else torch.as_tensor(resampled)

    new_meta = CTMeta(
        path=meta.path,
        affine=affine,
        spacing=tuple(float(s) for s in spacing),
        orientation=meta.orientation,
        original_shape=meta.original_shape,
    )
    return resampled, new_meta


def load_and_window(
    path: Union[str, os.PathLike],
    *,
    crop_size: Sequence[int] = DEFAULT_CROP_SIZE,
    spacing: Optional[Sequence[float]] = None,
    hu_range: Tuple[float, float] = DEFAULT_HU_RANGE,
    pad_short_axes: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, int, int], CTMeta]:
    """Load a NIfTI scan and turn it into the crops SPECTRE's backbone consumes.

    Applies the pretraining pipeline in order: load as RAS -> HU scale -> optional resample ->
    centre crop to a whole multiple of `crop_size` -> grid patch.

    The HU scaling deliberately happens *before* resampling, matching the eval scripts.
    Interpolating raw HU and then clipping is not the same as clipping and then interpolating -
    they diverge wherever bone (>1000 HU) meets soft tissue.

    Args:
        path: A .nii or .nii.gz file.
        crop_size: Spatial size (H, W, D) of one crop.
        spacing: Target voxel spacing in mm. None (the default) leaves the scan at its native
            spacing.
        hu_range: The HU window mapped onto [0, 1].
        pad_short_axes: Pad axes shorter than one crop instead of raising.

    Returns:
        The crops `(N, C, cH, cW, cD)`, the grid `(n_h, n_w, n_d)`, and the scan's metadata.
    """
    volume, meta = load_ct(path, orientation="RAS")

    if spacing is None:
        crops, grid = window_scan(
            volume, crop_size, hu_range=hu_range, pad_short_axes=pad_short_axes,
        )
        return crops, grid, meta

    volume = scale_intensity_range(volume, a_min=hu_range[0], a_max=hu_range[1])
    volume, meta = resample(volume, meta, spacing)
    crops, grid = window_scan(
        volume, crop_size, scale_intensity=False, pad_short_axes=pad_short_axes,
    )
    return crops, grid, meta
