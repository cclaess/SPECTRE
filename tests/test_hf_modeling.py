"""The `transformers` wrapper around SpectreImageFeatureExtractor.

Uses a small randomly-initialised preset: these tests are about the wrapper's call contract, not
embedding quality, so no weights are downloaded.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("transformers")

from transformers.modeling_outputs import BaseModelOutput  # noqa: E402


def _load(name):
    """Load an hf_export module without putting hf_export on sys.path.

    hf_export/ holds a generated `spectre/` copy from the last export run; adding it to sys.path
    would shadow the real package with that stale snapshot.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "hf_export" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SpectreConfig = _load("configuration_spectre").SpectreConfig
SpectreModel = _load("modeling_spectre").SpectreModel


@pytest.fixture(scope="module")
def model():
    return SpectreModel(SpectreConfig(preset="spectre-small")).eval()


@pytest.fixture(scope="module")
def crops():
    return torch.randn(1, 2, 1, 128, 128, 64)


GRID = (2, 1, 1)


def test_forward_returns_a_tensor_by_default(model, crops):
    with torch.no_grad():
        out = model(crops, grid_size=GRID)
    assert isinstance(out, torch.Tensor)


def test_return_dict_gives_a_model_output(model, crops):
    with torch.no_grad():
        out = model(crops, grid_size=GRID, return_dict=True)
    assert isinstance(out, BaseModelOutput)
    assert isinstance(out.last_hidden_state, torch.Tensor)


def test_raw_scan_is_windowed_internally(model):
    scan = torch.rand(1, 256, 128, 64) * 3000 - 1200
    with torch.no_grad():
        out = model(scan)
    assert out.ndim == 2


def test_crop_size_is_exposed(model):
    assert model.crop_size == (128, 128, 64)


# --- the call contract with transformers-style kwargs -----------------------------------------

@pytest.mark.parametrize("kwarg", ["use_cache", "output_norms", "some_future_kwarg"])
def test_unknown_transformers_kwargs_are_ignored(model, crops, kwarg):
    """Generic transformers callers pass these; they must not blow up a feature extractor.

    Regression: forwarding **kwargs into the extractor turned these into a TypeError.
    """
    with torch.no_grad():
        out = model(crops, grid_size=GRID, **{kwarg: False})
    assert isinstance(out, torch.Tensor)


def test_max_crops_per_forward_is_honoured(model, crops):
    """The one extractor kwarg worth exposing; it must reach the extractor, not the sink."""
    with torch.no_grad():
        whole = model(crops, grid_size=GRID)
        chunked = model(crops, grid_size=GRID, max_crops_per_forward=1)
    assert torch.allclose(whole, chunked, atol=1e-5)


def test_output_hidden_states_is_refused_not_ignored(model, crops):
    """Silently returning no hidden states would be worse than saying we cannot."""
    with pytest.raises(NotImplementedError, match="hidden states"):
        model(crops, grid_size=GRID, output_hidden_states=True)


def test_output_attentions_is_refused_not_ignored(model, crops):
    with pytest.raises(NotImplementedError, match="attention weights"):
        model(crops, grid_size=GRID, output_attentions=True)


@pytest.mark.parametrize("falsy", [False, None])
def test_falsy_output_flags_are_fine(model, crops, falsy):
    with torch.no_grad():
        out = model(crops, grid_size=GRID, output_hidden_states=falsy, output_attentions=falsy)
    assert isinstance(out, torch.Tensor)


def test_the_refusal_points_at_something_that_exists(model):
    """The error tells people to use forward_intermediates; make sure it is really there."""
    assert hasattr(model.model.backbone, "forward_intermediates")
