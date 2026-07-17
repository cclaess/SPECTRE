"""Parity between `spectre.windowing` (pure torch) and the MONAI pipeline used for pretraining.

A mismatch here does not raise anywhere in the model - it silently shifts every embedding - so
these tests gate the whole inference path.
"""
import numpy as np
import pytest
import torch

from spectre.windowing import (
    DEFAULT_CROP_SIZE,
    grid_patch,
    largest_multiple_center_crop,
    largest_multiple_crop_size,
    scale_intensity_range,
    window_scan,
)

monai = pytest.importorskip("monai", reason="MONAI is needed to compare against the reference pipeline")

from monai.transforms import (  # noqa: E402
    Compose,
    GridPatchd,
    ScaleIntensityRanged,
)

from spectre.transforms import LargestMultipleCenterCropd  # noqa: E402


def reference_pipeline(volume: torch.Tensor, crop_size):
    """The exact MONAI pipeline from eval/save_embeddings_ct_rate.py, minus IO and orientation."""
    transform = Compose([
        ScaleIntensityRanged(
            keys=("image",), a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True,
        ),
        LargestMultipleCenterCropd(keys=("image",), patch_size=crop_size),
        GridPatchd(keys=("image",), patch_size=crop_size, overlap=0.0),
    ])
    out = transform({"image": volume.clone()})["image"]
    return torch.as_tensor(np.asarray(out))


def hu_volume(*shape, seed=0):
    """A plausible CT volume in Hounsfield Units, including out-of-window bone and air."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, *shape, generator=g) * 3000.0 - 1200.0


@pytest.mark.parametrize("shape", [
    (256, 256, 128),   # exact multiple of the crop
    (300, 260, 130),   # ragged -> exercises the centre-crop offset
    (128, 128, 64),    # exactly one crop
    (400, 130, 200),   # very ragged
])
def test_window_scan_matches_monai(shape):
    volume = hu_volume(*shape)
    crops, grid = window_scan(volume, DEFAULT_CROP_SIZE)
    expected = reference_pipeline(volume, DEFAULT_CROP_SIZE)

    assert crops.shape == expected.shape
    assert torch.allclose(crops, expected, atol=0, rtol=0), "windowing diverged from MONAI"
    assert grid[0] * grid[1] * grid[2] == crops.shape[0]


def test_patch_order_is_depth_fastest():
    """Crop n must sit at grid position n = (h * n_w + w) * n_d + d, matching RoPE's coords."""
    crop = (2, 2, 2)
    # Each voxel holds its own flat index, so a misordered crop is immediately visible.
    volume = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(1, 4, 6, 8)
    crops, (n_h, n_w, n_d) = grid_patch(volume, crop)

    assert (n_h, n_w, n_d) == (2, 3, 4)
    for h in range(n_h):
        for w in range(n_w):
            for d in range(n_d):
                n = (h * n_w + w) * n_d + d
                expected = volume[
                    :,
                    h * crop[0]:(h + 1) * crop[0],
                    w * crop[1]:(w + 1) * crop[1],
                    d * crop[2]:(d + 1) * crop[2],
                ]
                assert torch.equal(crops[n], expected), f"crop {n} is not at grid ({h},{w},{d})"


def test_patch_order_matches_monai_directly():
    volume = hu_volume(256, 384, 128)
    crops, _ = window_scan(volume, DEFAULT_CROP_SIZE)
    expected = reference_pipeline(volume, DEFAULT_CROP_SIZE)
    # Asymmetric grid (2, 3, 2): a transposed order would still match in shape but not in content.
    for n in range(crops.shape[0]):
        assert torch.equal(crops[n], expected[n]), f"crop {n} differs from MONAI"


@pytest.mark.parametrize("shape,crop", [
    ((10, 10, 10), (5, 5, 5)),     # odd crop, even volume -> (S//2)-(R//2) != (S-R)//2
    ((100, 100, 100), (7, 7, 7)),
    ((64, 64, 64), (3, 3, 3)),
])
def test_center_crop_offset_matches_monai_for_odd_crops(shape, crop):
    volume = hu_volume(*shape)
    ours = largest_multiple_center_crop(volume, crop)
    reference = LargestMultipleCenterCropd(keys=("image",), patch_size=crop)({"image": volume.clone()})["image"]
    reference = torch.as_tensor(np.asarray(reference))
    assert ours.shape == reference.shape
    assert torch.equal(ours, reference), "centre-crop offset diverged from MONAI"


def test_center_crop_offset_is_not_naive_formula():
    """Pin the exact divergence, so a 'simplification' to (S - R) // 2 fails loudly.

    Axis of 10 with crop 3 -> roi 9 (odd, from an even axis). MONAI starts at
    (10 // 2) - (9 // 2) = 1; the naive (10 - 9) // 2 would start at 0.
    """
    volume = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1)
    cropped = largest_multiple_center_crop(volume, (3, 1, 1))
    assert cropped[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_scale_intensity_matches_monai():
    volume = hu_volume(32, 32, 16)
    ours = scale_intensity_range(volume)
    reference = ScaleIntensityRanged(
        keys=("image",), a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True,
    )({"image": volume.clone()})["image"]
    assert torch.allclose(ours, torch.as_tensor(np.asarray(reference)), atol=0, rtol=0)
    assert float(ours.min()) >= 0.0 and float(ours.max()) <= 1.0


def test_largest_multiple_crop_size_keeps_short_axis():
    assert largest_multiple_crop_size((300, 260, 130), (128, 128, 64)) == (256, 256, 128)
    assert largest_multiple_crop_size((100, 260, 130), (128, 128, 64)) == (100, 256, 128)


def test_short_axis_pads_with_air_and_warns():
    """Today this input crashes in torch.stack; it must now pad and say so."""
    volume = hu_volume(128, 128, 40)
    with pytest.warns(RuntimeWarning, match="smaller than one crop"):
        crops, grid = window_scan(volume, DEFAULT_CROP_SIZE)
    assert crops.shape == (1, 1, 128, 128, 64)
    assert grid == (1, 1, 1)
    # Padding is air (0.0 after scaling) and centred: 12 slices either side of 40.
    assert float(crops[0, 0, :, :, :12].max()) == 0.0
    assert float(crops[0, 0, :, :, -12:].max()) == 0.0


def test_short_axis_can_raise_instead():
    volume = hu_volume(128, 128, 40)
    with pytest.raises(ValueError, match="smaller than one crop"):
        window_scan(volume, DEFAULT_CROP_SIZE, pad_short_axes=False)


def test_non_hu_input_warns():
    already_scaled = torch.rand(1, 128, 128, 64)
    with pytest.warns(RuntimeWarning, match="does not look like Hounsfield Units"):
        window_scan(already_scaled, DEFAULT_CROP_SIZE)


def test_scale_intensity_false_skips_rescaling():
    already_scaled = torch.rand(1, 128, 128, 64)
    crops, _ = window_scan(already_scaled, DEFAULT_CROP_SIZE, scale_intensity=False)
    assert torch.equal(crops[0], already_scaled)


def test_accepts_unchanneled_volume():
    volume = hu_volume(128, 128, 64).squeeze(0)
    crops, grid = window_scan(volume, DEFAULT_CROP_SIZE)
    assert crops.shape == (1, 1, 128, 128, 64)
    assert grid == (1, 1, 1)


def test_grid_patch_rejects_ragged_volume():
    with pytest.raises(ValueError, match="not a whole multiple"):
        grid_patch(torch.zeros(1, 130, 128, 64), DEFAULT_CROP_SIZE)
