"""The pre-0.2 public API must keep working unchanged, and warn.

`MODEL_CONFIGS` is pinned against a golden copy of the original literal rather than derived from
`spectre.presets`, because the two disagree on what "spectre-large" means: here it is weightless,
while `from_pretrained("spectre-large")` loads weights.
"""
import warnings

import pytest
import torch

from spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor
from spectre.presets import PRESETS, get_preset, list_presets, list_pretrained

BACKBONE_URL = "https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128.pt?download=true"
COMBINER_URL = "https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_combiner_feature_vit_large.pt?download=true"

BACKBONE_KWARGS = {
    "num_classes": 0,
    "global_pool": '',
    "pos_embed": "rope",
    "rope_kwargs": {"base": 1000.0},
    "init_values": 1.0,
}
COMBINER_KWARGS = {
    "num_classes": 0,
    "global_pool": '',
    "pos_embed": "rope",
    "rope_kwargs": {"base": 100.0},
    "init_values": 1.0,
}

GOLDEN_MODEL_CONFIGS = {
    "spectre-small": {
        "name": "spectre-small",
        "backbone": "vit_small_patch16_128",
        "backbone_checkpoint_path_or_url": None,
        "backbone_kwargs": BACKBONE_KWARGS,
        "feature_combiner": "feat_vit_small",
        "feature_combiner_checkpoint_path_or_url": None,
        "feature_combiner_kwargs": COMBINER_KWARGS,
        "description": "SPECTRE model with ViT-Small backbone and feature combiner.",
    },
    "spectre-base": {
        "name": "spectre-base",
        "backbone": "vit_base_patch16_128",
        "backbone_checkpoint_path_or_url": None,
        "backbone_kwargs": BACKBONE_KWARGS,
        "feature_combiner": "feat_vit_base",
        "feature_combiner_checkpoint_path_or_url": None,
        "feature_combiner_kwargs": COMBINER_KWARGS,
        "description": "SPECTRE model with ViT-Base backbone and feature combiner.",
    },
    "spectre-large": {
        "name": "spectre-large",
        "backbone": "vit_large_patch16_128",
        "backbone_checkpoint_path_or_url": None,
        "backbone_kwargs": BACKBONE_KWARGS,
        "feature_combiner": "feat_vit_large",
        "feature_combiner_checkpoint_path_or_url": None,
        "feature_combiner_kwargs": COMBINER_KWARGS,
        "description": "SPECTRE model with ViT-Large backbone and feature combiner.",
    },
    "spectre-large-pretrained": {
        "name": "spectre-large-pretrained",
        "backbone": "vit_large_patch16_128",
        "backbone_checkpoint_path_or_url": BACKBONE_URL,
        "backbone_kwargs": BACKBONE_KWARGS,
        "feature_combiner": "feat_vit_large",
        "feature_combiner_checkpoint_path_or_url": COMBINER_URL,
        "feature_combiner_kwargs": COMBINER_KWARGS,
        "description": "Pretrained SPECTRE model with ViT-Large backbone and feature combiner.",
    },
}


def test_model_configs_matches_golden_copy():
    assert dict(MODEL_CONFIGS) == GOLDEN_MODEL_CONFIGS


def test_model_configs_getitem_warns():
    with pytest.warns(FutureWarning, match="MODEL_CONFIGS is deprecated"):
        config = MODEL_CONFIGS["spectre-large-pretrained"]
    assert config["backbone"] == "vit_large_patch16_128"


def test_model_configs_get_warns():
    """dict.get does not route through __getitem__, so it needs its own warning."""
    with pytest.warns(FutureWarning, match="MODEL_CONFIGS is deprecated"):
        config = MODEL_CONFIGS.get("spectre-large-pretrained")
    assert config is not None


def test_from_config_warns_and_still_builds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = dict(MODEL_CONFIGS["spectre-small"])
    with pytest.warns(FutureWarning, match="from_config.*is deprecated"):
        model = SpectreImageFeatureExtractor.from_config(config)
    assert model.backbone is not None
    assert model.feature_combiner is not None


def test_hub_constructor_signature_still_works():
    """hf_export/modeling_spectre.py calls exactly this. Breaking it breaks the published model."""
    model = SpectreImageFeatureExtractor(
        backbone_name="vit_small_patch16_128",
        backbone_kwargs=BACKBONE_KWARGS,
        feature_combiner_name="feat_vit_small",
        feature_combiner_kwargs=COMBINER_KWARGS,
    )
    assert model.backbone is not None
    assert model.feature_combiner is not None


def test_importing_spectre_does_not_warn():
    """A module-level warning would fire inside every Hub trust_remote_code import."""
    import importlib

    import spectre
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        importlib.reload(spectre)


def test_legacy_extract_and_combine_still_work():
    model = SpectreImageFeatureExtractor.from_pretrained("spectre-small", pretrained=False).eval()
    crops = torch.rand(1, 2, 1, 128, 128, 64)
    with torch.no_grad():
        features = model.extract_backbone_features(crops)
        combined = model.combine_features(features, (2, 1, 1))
    assert features.ndim == 4
    assert combined.ndim == 3


def test_mutable_default_kwargs_are_not_shared():
    """The old signature used dict literals as defaults, which leak between instances."""
    a = SpectreImageFeatureExtractor(backbone_name="vit_small_patch16_128")
    b = SpectreImageFeatureExtractor(backbone_name="vit_small_patch16_128")
    assert a.backbone is not b.backbone


# --- presets ----------------------------------------------------------------------------------

def test_list_presets_keeps_all_sizes():
    assert list_presets() == ["spectre-small", "spectre-base", "spectre-large"]


def test_only_large_has_weights():
    assert list_pretrained() == ["spectre-large"]


def test_get_preset_returns_deep_copies():
    """Mutating a returned preset must not corrupt the registry for the rest of the process."""
    preset = get_preset("spectre-large")
    preset.backbone_kwargs["rope_kwargs"]["base"] = 999.0
    assert PRESETS["spectre-large"].backbone_kwargs["rope_kwargs"]["base"] == 1000.0
    assert get_preset("spectre-large").backbone_kwargs["rope_kwargs"]["base"] == 1000.0


def test_unknown_preset_suggests_a_close_match():
    with pytest.raises(ValueError, match="Did you mean 'spectre-large'"):
        get_preset("spectre-larg")


def test_unknown_preset_lists_options():
    with pytest.raises(ValueError, match="Available:"):
        get_preset("nonsense")


def test_pretrained_on_weightless_preset_raises_rather_than_random_init():
    with pytest.raises(ValueError, match="no published weights"):
        SpectreImageFeatureExtractor.from_pretrained("spectre-small", pretrained=True)


def test_from_pretrained_rejects_a_path():
    with pytest.raises(ValueError, match="preset name, not a path"):
        SpectreImageFeatureExtractor.from_pretrained("/tmp/my_checkpoint.pt")


def test_from_pretrained_returns_eval_mode():
    model = SpectreImageFeatureExtractor.from_pretrained("spectre-small", pretrained=False)
    assert model.training is False


def test_backbone_only_skips_the_combiner():
    model = SpectreImageFeatureExtractor.from_pretrained(
        "spectre-small", pretrained=False, include_feature_combiner=False,
    )
    assert model.feature_combiner is None
    assert model.has_feature_combiner is False


def test_kwargs_overrides_are_merged_over_the_preset():
    model = SpectreImageFeatureExtractor.from_pretrained(
        "spectre-small", pretrained=False, backbone_kwargs={"init_values": None},
    )
    # init_values=None swaps LayerScale for Identity, so this is observable on the module tree.
    assert not hasattr(model.backbone.blocks[0].ls1, "gamma")
