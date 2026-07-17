"""Reading and resampling CT scans (`spectre.io`).

The resampling path had no coverage, which is how a call to a nonexistent `Spacing(affine=...)`
argument shipped. These exercise it against real NIfTI files.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("nibabel", reason="spectre.io needs the [inference] extra")
pytest.importorskip("monai", reason="resampling needs MONAI")
import nibabel as nib  # noqa: E402

from spectre.io import CTMeta, load_and_window, load_ct, resample  # noqa: E402

SPACING = (0.7, 0.7, 1.5)


def write_scan(path, shape=(60, 50, 40), spacing=SPACING, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) * 3000 - 1200).astype(np.float32)
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def test_load_ct_reports_native_spacing(tmp_path):
    scan = write_scan(tmp_path / "scan.nii.gz")
    volume, meta = load_ct(scan)
    assert volume.shape[0] == 1  # channel-first
    assert isinstance(meta, CTMeta)
    assert meta.spacing == pytest.approx(SPACING, abs=1e-4)
    assert meta.orientation == "RAS"


def test_resample_changes_shape_and_spacing(tmp_path):
    """The regression: this used to raise TypeError on Spacing(affine=...)."""
    scan = write_scan(tmp_path / "scan.nii.gz", shape=(60, 50, 40))
    volume, meta = load_ct(scan)

    resampled, new_meta = resample(volume, meta, (0.5, 0.5, 1.0))

    assert new_meta.spacing == (0.5, 0.5, 1.0)
    # Finer spacing on two axes -> more voxels; coarser depth spacing (1.0 < 1.5) -> more too.
    assert resampled.shape[1] > volume.shape[1]
    assert resampled.shape[2] > volume.shape[2]
    assert torch.isfinite(resampled).all()


def test_resample_to_native_spacing_is_close_to_identity(tmp_path):
    scan = write_scan(tmp_path / "scan.nii.gz")
    volume, meta = load_ct(scan)
    resampled, _ = resample(volume, meta, SPACING)
    assert tuple(resampled.shape) == tuple(volume.shape)


def test_resample_scale_factor_matches_spacing_ratio(tmp_path):
    scan = write_scan(tmp_path / "scan.nii.gz", shape=(60, 60, 60), spacing=(1.0, 1.0, 1.0))
    volume, meta = load_ct(scan)
    resampled, _ = resample(volume, meta, (0.5, 0.5, 0.5))
    # Halving the spacing doubles each axis (allow a voxel of rounding).
    for axis in (1, 2, 3):
        assert resampled.shape[axis] == pytest.approx(2 * volume.shape[axis], abs=1)


def test_resample_returns_a_plain_tensor(tmp_path):
    """Downstream windowing expects a torch.Tensor, not a MetaTensor."""
    scan = write_scan(tmp_path / "scan.nii.gz")
    volume, meta = load_ct(scan)
    resampled, _ = resample(volume, meta, (0.5, 0.5, 1.0))
    assert type(resampled) is torch.Tensor


def test_resample_rejects_wrong_length_spacing(tmp_path):
    scan = write_scan(tmp_path / "scan.nii.gz")
    volume, meta = load_ct(scan)
    with pytest.raises(ValueError, match="3 elements"):
        resample(volume, meta, (0.5, 0.5))


def test_load_and_window_without_spacing_keeps_native(tmp_path):
    scan = write_scan(tmp_path / "scan.nii.gz", shape=(256, 128, 64))
    crops, grid, meta = load_and_window(scan)
    assert crops.shape[1:] == (1, 128, 128, 64)
    assert grid == (2, 1, 1)
    assert meta.spacing == pytest.approx(SPACING, abs=1e-4)


def test_load_and_window_with_spacing_resamples_then_crops(tmp_path):
    """The full CLI --spacing path: load -> scale -> resample -> crop -> patch."""
    scan = write_scan(tmp_path / "scan.nii.gz", shape=(200, 200, 100), spacing=(1.0, 1.0, 1.0))
    crops, grid, _ = load_and_window(scan, spacing=(0.5, 0.5, 1.0))
    assert crops.shape[1:] == (1, 128, 128, 64)
    # 200 @ 1.0 -> 400 @ 0.5 gives 3 crops of 128 across H and W; depth 100 stays 100 -> 1 crop.
    assert grid == (3, 3, 1)
    assert torch.isfinite(crops).all()


def test_load_and_window_scales_before_resampling(tmp_path):
    """Intensity must already be in [0, 1] after windowing, even on the resample path."""
    scan = write_scan(tmp_path / "scan.nii.gz", shape=(140, 140, 70), spacing=(1.0, 1.0, 1.0))
    crops, _, _ = load_and_window(scan, spacing=(0.8, 0.8, 1.0))
    # Interpolation can overshoot slightly past the clamp, but not by much.
    assert float(crops.min()) >= -0.05
    assert float(crops.max()) <= 1.05
