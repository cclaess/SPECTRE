"""The Hugging Face SpectreConfig.

The published `config.json` and the vendored code ship and are replaced together, so the schema
is free to change - but a config saved by an older release must still load its own architecture
rather than silently inheriting this class's defaults. That silent-fallback case is the one that
would quietly serve the wrong model, so it is pinned here.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("transformers")

from spectre.presets import BACKBONE_KWARGS, COMBINER_KWARGS, get_preset  # noqa: E402


def _load_configuration_spectre():
    """Load hf_export/configuration_spectre.py without putting hf_export on sys.path.

    hf_export/ contains a generated `spectre/` copy from the last export run; adding it to
    sys.path would shadow the real package with that stale snapshot.
    """
    path = ROOT / "hf_export" / "configuration_spectre.py"
    spec = importlib.util.spec_from_file_location("_hf_configuration_spectre", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpectreConfig


SpectreConfig = _load_configuration_spectre()

# Exactly what a pre-0.2 config.json carries: no preset, no preprocessing fields.
LEGACY_CONFIG = {
    "backbone_name": "vit_large_patch16_128",
    "backbone_kwargs": {
        "num_classes": 0, "global_pool": "", "pos_embed": "rope",
        "rope_kwargs": {"base": 1000.0}, "init_values": 1.0,
    },
    "feature_combiner_name": "feat_vit_large",
    "feature_combiner_kwargs": {
        "num_classes": 0, "global_pool": "", "pos_embed": "rope",
        "rope_kwargs": {"base": 100.0}, "init_values": 1.0,
    },
}


def test_defaults_resolve_from_the_preset_registry():
    config = SpectreConfig()
    preset = get_preset("spectre-large")
    assert config.preset == "spectre-large"
    assert config.backbone_name == preset.backbone
    assert config.feature_combiner_name == preset.feature_combiner
    assert config.backbone_kwargs == BACKBONE_KWARGS
    assert config.feature_combiner_kwargs == COMBINER_KWARGS


def test_config_states_the_preprocessing_contract():
    """The point of the rewrite: config.json should say what the model expects to be fed."""
    config = SpectreConfig()
    assert config.crop_size == (128, 128, 64)
    assert config.hu_range == (-1000.0, 1000.0)
    assert config.voxel_spacing is None  # native spacing
    assert config.orientation == "RAS"


def test_nothing_about_the_model_is_restated_in_this_module():
    """Architecture must come from spectre.presets, preprocessing from spectre.windowing."""
    source = (ROOT / "hf_export" / "configuration_spectre.py").read_text(encoding="utf-8")
    for magic in ("vit_large_patch16_128", "feat_vit_large", "rope", "init_values",
                  "global_pool", "(128, 128, 64)"):
        assert magic not in source, (
            f"{magic!r} is restated in configuration_spectre.py; import it instead"
        )


def test_preprocessing_defaults_track_the_package():
    """crop_size/hu_range must be the same objects the package windows scans with."""
    from spectre.windowing import DEFAULT_CROP_SIZE, DEFAULT_HU_RANGE

    config = SpectreConfig()
    assert config.crop_size == DEFAULT_CROP_SIZE
    assert config.hu_range == DEFAULT_HU_RANGE


def test_other_presets_resolve():
    config = SpectreConfig(preset="spectre-small")
    assert config.backbone_name == "vit_small_patch16_128"
    assert config.feature_combiner_name == "feat_vit_small"


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown SPECTRE preset"):
        SpectreConfig(preset="spectre-huge")


def test_backbone_only_drops_the_combiner():
    config = SpectreConfig(include_feature_combiner=False)
    assert config.feature_combiner_name is None


# --- compatibility, both directions -----------------------------------------------------------

def test_legacy_config_json_keeps_its_own_architecture():
    """The dangerous case: explicit values must win, never fall back to defaults."""
    config = SpectreConfig(**LEGACY_CONFIG)
    assert config.backbone_name == "vit_large_patch16_128"
    assert config.backbone_kwargs["rope_kwargs"]["base"] == 1000.0
    assert config.feature_combiner_kwargs["rope_kwargs"]["base"] == 100.0


def test_legacy_config_for_a_different_size_is_not_overridden_by_large_defaults():
    """A stale small config must not silently load the Large architecture."""
    small = dict(LEGACY_CONFIG, backbone_name="vit_small_patch16_128",
                 feature_combiner_name="feat_vit_small")
    config = SpectreConfig(**small)
    assert config.backbone_name == "vit_small_patch16_128"
    assert config.feature_combiner_name == "feat_vit_small"


def test_new_config_round_trips_through_json():
    config = SpectreConfig()
    restored = SpectreConfig.from_dict(config.to_dict())
    assert restored.backbone_name == config.backbone_name
    assert restored.preset == config.preset
    assert tuple(restored.crop_size) == config.crop_size
    assert tuple(restored.hu_range) == config.hu_range


def test_serialised_config_still_carries_the_resolved_architecture():
    """Older code reads backbone_name/kwargs and ignores the new fields; keep them present."""
    payload = SpectreConfig().to_dict()
    for key in ("backbone_name", "backbone_kwargs", "feature_combiner_name",
                "feature_combiner_kwargs"):
        assert key in payload, f"{key} must stay in config.json for older readers"
    assert payload["backbone_name"] == "vit_large_patch16_128"


def test_save_and_reload(tmp_path):
    SpectreConfig().save_pretrained(tmp_path)
    reloaded = SpectreConfig.from_pretrained(tmp_path)
    assert reloaded.backbone_name == "vit_large_patch16_128"
    assert reloaded.preset == "spectre-large"
    assert tuple(reloaded.crop_size) == (128, 128, 64)


def test_explicit_kwargs_override_the_preset():
    config = SpectreConfig(preset="spectre-large", backbone_name="vit_small_patch16_128")
    assert config.backbone_name == "vit_small_patch16_128"
