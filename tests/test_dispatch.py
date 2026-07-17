"""Input dispatch, batching and equivalence for `SpectreImageFeatureExtractor.forward`.

Uses a tiny randomly-initialised model: these tests are about shapes, dispatch and the
equivalence of the batched path to per-scan calls, not about embedding quality.
"""
import pytest
import torch

from spectre import SpectreImageFeatureExtractor
from spectre.windowing import window_scan

CROP = (128, 128, 64)


@pytest.fixture(scope="module")
def model():
    m = SpectreImageFeatureExtractor.from_pretrained(
        "spectre-small", pretrained=False, include_feature_combiner=True,
    )
    return m.eval()


@pytest.fixture(scope="module")
def backbone_only():
    m = SpectreImageFeatureExtractor.from_pretrained(
        "spectre-small", pretrained=False, include_feature_combiner=False,
    )
    return m.eval()


def hu_scan(h, w, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, d, generator=g) * 3000.0 - 1200.0


def test_crop_size_comes_from_backbone(model):
    assert model.crop_size == CROP


def test_has_feature_combiner_flag(model, backbone_only):
    assert model.has_feature_combiner is True
    assert backbone_only.has_feature_combiner is False


# --- dispatch table ---------------------------------------------------------------------------

def test_3d_raw_scan(model):
    with torch.no_grad():
        out = model(hu_scan(128, 128, 64).squeeze(0))
    assert out.ndim == 2  # (T', F')


def test_4d_raw_scan(model):
    with torch.no_grad():
        out = model(hu_scan(256, 128, 64))
    assert out.ndim == 2


def test_5d_prewindowed_scan(model):
    crops, grid = window_scan(hu_scan(256, 128, 64), CROP)
    with torch.no_grad():
        out = model(crops, grid_size=grid)
    assert out.ndim == 2


def test_6d_legacy_batch(model):
    crops_a, grid = window_scan(hu_scan(256, 128, 64, seed=1), CROP)
    crops_b, _ = window_scan(hu_scan(256, 128, 64, seed=2), CROP)
    batch = torch.stack([crops_a, crops_b], dim=0)
    with torch.no_grad():
        out = model(batch, grid_size=grid)
    assert out.shape[0] == 2 and out.ndim == 3  # (B, T', F')


def test_list_of_raw_scans_unequal_sizes(model):
    scans = [hu_scan(256, 128, 64, seed=1), hu_scan(128, 128, 128, seed=2), hu_scan(384, 256, 64, seed=3)]
    with torch.no_grad():
        out = model(scans)
    assert isinstance(out, list) and len(out) == 3
    assert all(o.ndim == 2 for o in out)


def test_backbone_only_shapes(backbone_only):
    with torch.no_grad():
        out = backbone_only(hu_scan(256, 128, 64))
    assert out.ndim == 3  # (N, T, F)
    assert out.shape[0] == 2  # 2 crops


def test_backbone_only_list(backbone_only):
    scans = [hu_scan(256, 128, 64, seed=1), hu_scan(128, 128, 128, seed=2)]
    with torch.no_grad():
        out = backbone_only(scans)
    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape[0] == 2 and out[1].shape[0] == 2


# --- disambiguation guards --------------------------------------------------------------------

def test_5d_with_wrong_crop_size_is_rejected(model):
    """A raw same-size batch (B, C, H, W, D) must not be silently read as one scan's crops."""
    raw_batch = torch.rand(4, 1, 256, 256, 128)
    with pytest.raises(ValueError, match="pre-windowed scan"):
        model(raw_batch, grid_size=(1, 1, 1))


def test_5d_without_grid_size_is_rejected(model):
    crops, _ = window_scan(hu_scan(256, 128, 64), CROP)
    with pytest.raises(ValueError, match="grid_size is required"):
        model(crops)


def test_grid_size_with_raw_scan_is_rejected(model):
    with pytest.raises(ValueError, match="must not be passed for a raw"):
        model(hu_scan(128, 128, 64), grid_size=(1, 1, 1))


def test_mismatched_grid_size_is_rejected(model):
    crops, _ = window_scan(hu_scan(256, 128, 64), CROP)  # 2 crops
    with pytest.raises(ValueError, match="implies"):
        model(crops, grid_size=(3, 3, 3))


def test_unsupported_rank_is_rejected(model):
    with pytest.raises(ValueError, match="Unsupported input"):
        model(torch.rand(4, 4))


def test_wrong_channel_count_is_rejected(model):
    with pytest.raises(ValueError, match="one raw scan"):
        model(torch.rand(3, 128, 128, 64) * 1000 - 1000)


# --- equivalences -----------------------------------------------------------------------------

def test_batched_list_matches_per_scan_calls(model):
    scans = [hu_scan(256, 128, 64, seed=1), hu_scan(128, 128, 128, seed=2), hu_scan(256, 128, 64, seed=3)]
    with torch.no_grad():
        batched = model(scans)
        individual = [model(s) for s in scans]
    for b, i in zip(batched, individual):
        assert torch.allclose(b, i, atol=1e-5), "batched path diverged from per-scan calls"


def test_grid_grouping_preserves_order(model):
    """Scans 0 and 2 share a grid and are combined together; results must not be swapped."""
    scans = [hu_scan(256, 128, 64, seed=1), hu_scan(128, 128, 128, seed=2), hu_scan(256, 128, 64, seed=3)]
    with torch.no_grad():
        batched = model(scans)
        alone = [model(s) for s in scans]
    assert torch.allclose(batched[0], alone[0], atol=1e-5)
    assert torch.allclose(batched[2], alone[2], atol=1e-5)
    assert not torch.allclose(batched[0], batched[2], atol=1e-5), "distinct scans gave identical output"


def test_max_crops_per_forward_does_not_change_result(model):
    scans = [hu_scan(384, 256, 128, seed=1), hu_scan(256, 128, 64, seed=2)]
    with torch.no_grad():
        whole = model(scans, max_crops_per_forward=None)
        chunked = model(scans, max_crops_per_forward=3)
    for w, c in zip(whole, chunked):
        assert torch.allclose(w, c, atol=1e-5), "chunking changed the result"


def test_raw_and_prewindowed_agree(model):
    """Windowing internally must equal windowing outside and passing crops in."""
    scan = hu_scan(256, 128, 64, seed=7)
    crops, grid = window_scan(scan, CROP)
    with torch.no_grad():
        from_raw = model(scan)
        from_crops = model(crops, grid_size=grid)
    assert torch.allclose(from_raw, from_crops, atol=1e-6)


def test_forward_matches_legacy_6d_api(model):
    """The new dispatch must reproduce the old extract+combine path exactly."""
    crops_a, grid = window_scan(hu_scan(256, 128, 64, seed=1), CROP)
    crops_b, _ = window_scan(hu_scan(256, 128, 64, seed=2), CROP)
    batch = torch.stack([crops_a, crops_b], dim=0)
    with torch.no_grad():
        new = model(batch, grid_size=grid)
        legacy = model.combine_features(model.extract_backbone_features(batch), grid)
    assert torch.allclose(new, legacy, atol=1e-6), "forward diverged from extract+combine"


def test_extract_is_inference_mode(model):
    out = model.extract(hu_scan(128, 128, 64))
    assert out.requires_grad is False


def test_list_grid_size_broadcast_and_per_scan(model):
    a, grid = window_scan(hu_scan(256, 128, 64, seed=1), CROP)
    b, _ = window_scan(hu_scan(256, 128, 64, seed=2), CROP)
    with torch.no_grad():
        shared = model([a, b], grid_size=grid)             # one grid for both
        per_scan = model([a, b], grid_size=[grid, grid])   # one grid each
    for s, p in zip(shared, per_scan):
        assert torch.allclose(s, p, atol=1e-6)


def test_wrong_number_of_grids_is_rejected(model):
    a, grid = window_scan(hu_scan(256, 128, 64, seed=1), CROP)
    b, _ = window_scan(hu_scan(256, 128, 64, seed=2), CROP)
    with pytest.raises(ValueError, match="grid sizes for"):
        model([a, b], grid_size=[grid, grid, grid])
