"""The feature combiner must honour per-sample RoPE coordinate augmentation.

`FeatureVisionTransformer` used to build its RotaryPositionEmbedding without ever setting
`requires_per_sample_rope`, unlike `VisionTransformer`. The augmentation still fired (it is gated
on `self.training` inside the rope), but `_pos_embed` called the rope once for the whole batch,
so every sample got the *same* random shift/jitter/rescale instead of an independent one.
`pretrain_siglip.py` configures `rescale_coords=2.0` for both the backbone and the combiner, so
this silently weakened the combiner's augmentation during pretraining.
"""
import pytest
import torch

from spectre.models import FeatureVisionTransformer, vit_small_patch16_128

# 3D RoPE requires (embed_dim // num_heads) % 6 == 0.
COMBINER_KWARGS = dict(
    patch_dim=768, num_classes=0, global_pool='', pos_embed='rope',
    init_values=1.0, embed_dim=396, depth=2, num_heads=6,
)


def combiner(**rope_kwargs):
    return FeatureVisionTransformer(rope_kwargs={"base": 100.0, **rope_kwargs}, **COMBINER_KWARGS)


@pytest.mark.parametrize("aug", ["shift_coords", "jitter_coords", "rescale_coords"])
def test_each_augmentation_enables_per_sample_rope(aug):
    assert combiner(**{aug: 2.0}).requires_per_sample_rope is True


def test_no_augmentation_means_no_per_sample_rope():
    assert combiner().requires_per_sample_rope is False


def test_combiner_matches_backbone_behaviour():
    """The two classes must agree; they disagreed for exactly this reason."""
    rope_kwargs = {"base": 100.0, "rescale_coords": 2.0}
    backbone = vit_small_patch16_128(
        num_classes=0, global_pool='', pos_embed='rope', rope_kwargs=rope_kwargs, init_values=1.0,
    )
    assert combiner(rescale_coords=2.0).requires_per_sample_rope == backbone.requires_per_sample_rope


def test_training_gives_each_sample_its_own_rope():
    model = combiner(rescale_coords=2.0).train()
    torch.manual_seed(0)
    _, rope = model._pos_embed(torch.randn(4, 8, 396), grid_size=(2, 2, 2))

    assert isinstance(rope, list) and len(rope) == 4
    sins = [r[0] for r in rope]
    # All four must differ; previously all four were one shared tensor.
    assert len({s.sum().item() for s in sins}) == 4, "samples share a rope augmentation"


def test_eval_is_deterministic_and_unaugmented():
    """Inference must not change: the augmentations are gated on self.training."""
    model = combiner(rescale_coords=2.0).eval()
    _, first = model._pos_embed(torch.randn(4, 8, 396), grid_size=(2, 2, 2))
    _, second = model._pos_embed(torch.randn(4, 8, 396), grid_size=(2, 2, 2))

    a = torch.stack([r[0] for r in first])
    b = torch.stack([r[0] for r in second])
    assert torch.equal(a, b), "rope must be deterministic at eval"
    assert torch.equal(a[0], a[1]), "rope must be identical across samples at eval"


def test_published_preset_inference_is_unaffected():
    """The shipped presets configure no coord augmentation, so nothing about them changes."""
    from spectre.presets import get_preset

    preset = get_preset("spectre-large")
    for kwargs in (preset.backbone_kwargs, preset.feature_combiner_kwargs):
        assert set(kwargs["rope_kwargs"]) == {"base"}, (
            "a preset gained a coord augmentation; published embeddings would move"
        )


def test_forward_runs_with_per_sample_rope():
    """attention.py stacks the per-sample rope list; make sure that path actually executes."""
    model = combiner(rescale_coords=2.0).train()
    out = model(torch.randn(4, 8, 768), grid_size=(2, 2, 2))
    assert out.shape[0] == 4
    assert torch.isfinite(out).all()
