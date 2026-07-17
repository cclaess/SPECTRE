"""Named SPECTRE architectures and their published weights.

This module is vendored into the Hugging Face Hub repository by `scripts/export_hf.py`, so it
must import nothing beyond the standard library.
"""
from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    'SpectrePreset',
    'PRESETS',
    'BACKBONE_KWARGS',
    'COMBINER_KWARGS',
    'get_preset',
    'list_presets',
    'list_pretrained',
]

_HF_WEIGHTS = "https://huggingface.co/cclaess/SPECTRE/resolve/main/{}?download=true"

# The only place the SPECTRE architecture kwargs are written down. `spectre.model`, the Hugging
# Face SpectreConfig and scripts/export_hf.py all resolve back to here rather than restating them.
# The backbone sees raw CT crops; the feature combiner sees the backbone's per-crop embeddings,
# hence the different RoPE bases.
BACKBONE_KWARGS: Dict[str, Any] = {
    "num_classes": 0,
    "global_pool": '',
    "pos_embed": "rope",
    "rope_kwargs": {"base": 1000.0},
    "init_values": 1.0,
}
COMBINER_KWARGS: Dict[str, Any] = {
    "num_classes": 0,
    "global_pool": '',
    "pos_embed": "rope",
    "rope_kwargs": {"base": 100.0},
    "init_values": 1.0,
}


@dataclass(frozen=True)
class SpectrePreset:
    """One named SPECTRE architecture, optionally with published weights.

    There is deliberately no crop size here: it is `backbone.patch_embed.img_size`, so the
    architecture stays the single source of truth.
    """
    name: str
    backbone: str
    feature_combiner: str
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)
    feature_combiner_kwargs: Dict[str, Any] = field(default_factory=dict)
    backbone_weights: Optional[str] = None
    feature_combiner_weights: Optional[str] = None
    description: str = ""

    @property
    def has_pretrained_weights(self) -> bool:
        return self.backbone_weights is not None


PRESETS: Dict[str, SpectrePreset] = {
    "spectre-small": SpectrePreset(
        name="spectre-small",
        backbone="vit_small_patch16_128",
        feature_combiner="feat_vit_small",
        backbone_kwargs=copy.deepcopy(BACKBONE_KWARGS),
        feature_combiner_kwargs=copy.deepcopy(COMBINER_KWARGS),
        description="SPECTRE with a ViT-Small backbone. No published weights yet.",
    ),
    "spectre-base": SpectrePreset(
        name="spectre-base",
        backbone="vit_base_patch16_128",
        feature_combiner="feat_vit_base",
        backbone_kwargs=copy.deepcopy(BACKBONE_KWARGS),
        feature_combiner_kwargs=copy.deepcopy(COMBINER_KWARGS),
        description="SPECTRE with a ViT-Base backbone. No published weights yet.",
    ),
    "spectre-large": SpectrePreset(
        name="spectre-large",
        backbone="vit_large_patch16_128",
        feature_combiner="feat_vit_large",
        backbone_kwargs=copy.deepcopy(BACKBONE_KWARGS),
        feature_combiner_kwargs=copy.deepcopy(COMBINER_KWARGS),
        backbone_weights=_HF_WEIGHTS.format("spectre_backbone_vit_large_patch16_128.pt"),
        feature_combiner_weights=_HF_WEIGHTS.format("spectre_combiner_feature_vit_large.pt"),
        description="SPECTRE with a ViT-Large backbone, pretrained with SSL + vision-language alignment.",
    ),
}


def list_presets() -> List[str]:
    """Every known architecture name."""
    return list(PRESETS)


def list_pretrained() -> List[str]:
    """The architecture names that have published weights."""
    return [name for name, preset in PRESETS.items() if preset.has_pretrained_weights]


def get_preset(name: str) -> SpectrePreset:
    """Look up a preset by name.

    The returned preset owns deep copies of its kwargs, so callers can mutate them without
    corrupting the registry for the rest of the process.
    """
    try:
        preset = PRESETS[name]
    except KeyError:
        suggestion = difflib.get_close_matches(str(name), list(PRESETS), n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(
            f"Unknown SPECTRE preset {name!r}. Available: {list_presets()}.{hint}"
        ) from None

    return SpectrePreset(
        name=preset.name,
        backbone=preset.backbone,
        feature_combiner=preset.feature_combiner,
        backbone_kwargs=copy.deepcopy(preset.backbone_kwargs),
        feature_combiner_kwargs=copy.deepcopy(preset.feature_combiner_kwargs),
        backbone_weights=preset.backbone_weights,
        feature_combiner_weights=preset.feature_combiner_weights,
        description=preset.description,
    )
