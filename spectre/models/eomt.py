# Adapted from https://github.com/tue-mps/eomt/
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectre.models.layers import LayerNorm3d


class ScaleBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        scale_factors: Union[int, Tuple[int, int, int]] = (2, 2, 2),
        conv1_layer: nn.Module = nn.ConvTranspose3d,
    ):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=scale_factors,
            stride=scale_factors,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm3d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


def compute_upscale_stages(patch_size, min_size=4):
    # Compute how many times to upscale per dimension
    num_stages = []
    for size in patch_size:
        stages = max(0, int(math.log2(size)) - int(math.log2(min_size)))
        num_stages.append(stages)
    return num_stages


class EoMT(nn.Module):
    def __init__(
        self,
        backbone: "VisionTransformer",
        num_classes: int,
        num_q: int,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.backbone.embed_dim)

        self.class_head = nn.Linear(self.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim),
        )

        patch_size = self.backbone.patch_embed.patch_size
        num_upscale_stages = compute_upscale_stages(patch_size, min_size=4)

        # Build per-stage scale factors list
        max_stages = max(num_upscale_stages)
        upscale_blocks = []
        for stage_idx in range(max_stages):
            # for each dimension, upscale by 2 only if this dimension still has
            # remaining upscales at this stage
            scale_factors = tuple(
                2 if stage_idx < num_upscale_stages[dim] else 1
                for dim in range(len(patch_size))
            )
            upscale_blocks.append(ScaleBlock(self.backbone.embed_dim,
                                            scale_factors=scale_factors))

        self.upscale = nn.Sequential(*upscale_blocks)

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchwd -> bqhwd", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(self, module: 'Attention', x: torch.Tensor, mask: Optional[torch.Tensor], rope = None):
        B, N, C = x.shape

        q = module.q(x).reshape(B, N, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
        kv = module.kv(x).reshape(B, N, 2, module.num_heads, module.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if rope is not None:
            if isinstance(rope, list):
                rope = tuple(torch.stack([r[i] for r in rope], dim=0) for i in range(2))
            q, k = module.apply_rotary_pos_emb(q, k, rope)

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def forward(self, x: torch.Tensor):
        x = self.backbone.patch_embed(x)
        x, rope = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.backbone.blocks):
            if i == len(self.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                interpolated = F.interpolate(
                    mask_logits,
                    self.backbone.patch_embed.grid_size,
                    mode="trilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :,
                    : self.num_q,
                    self.num_q + self.backbone.num_prefix_tokens :,
                ] = (
                    interpolated > 0
                )
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[
                        i - len(self.backbone.blocks) + self.num_blocks
                    ],
                )

            x = x + block.drop_path1(
                block.ls1(self._attn(block.attn, block.norm1(x), attn_mask, rope))
            )
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        mask_logits, class_logits = self._predict(self.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
