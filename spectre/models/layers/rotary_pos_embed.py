import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
import numpy as np


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x:   [..., D], split into halves and rotate
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(
    x: torch.Tensor, 
    sin: torch.Tensor, 
    cos: torch.Tensor
) -> torch.Tensor:
    # x, sin, cos: [..., D]
    return (x * cos) + (rope_rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    """
    3D Rotary Positional Embedding (RoPE) with no mixing across axes (axial),
    and no learnable weights. Allows for shifting and scaling of the positional encodings
    for improving performance on varying resolutions.
    Mirrors DINOv3 style but for (H, W, D).

    Requirements:
      - head_dim % 6 == 0  (because 3 axes -> periods of size head_dim//6, then we tile to fill head_dim)

    Two parametrizations:
      * base
      * min_period + max_period
    """
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 1000.0,  # works for common 8^3=512 to 16^3=4096 tokens
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads
        assert head_dim % 6 == 0, "For 3D RoPE, (embed_dim // num_heads) must be divisible by 6"

        both_periods = (min_period is not None) and (max_period is not None)
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.head_dim = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Keep dtype persistent so teacher can be initialized from student state_dict()
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(self.head_dim // 6, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            # powers from 0..(head_dim // 3 - 1), normalized to [0, 1) across head_dim // 3? 
            # for 3D we use // 6 per axis
            periods = self.base ** (
                2 * torch.arange(self.head_dim // 6, device=device, dtype=dtype) / (self.head_dim // 3)
            )
        else:
            # geometric spacing between min_period and max_period
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.head_dim // 6, device=device, dtype=dtype)
            periods = base ** exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods

    def forward(self, *, H: int, W: int, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          sin, cos: [H * W * D, head_dim]   (per-head)
        """
        device = self.periods.device
        dtype = self.dtype
        dd = dict(device=device, dtype=dtype)

        # Prepare coords in [0, 1] then map to [-1, +1]
        if self.normalize_coords == "max":
            max_dim = max(H, W, D)
            coords_h = torch.arange(0.5, H, **dd) / max_dim
            coords_w = torch.arange(0.5, W, **dd) / max_dim
            coords_d = torch.arange(0.5, D, **dd) / max_dim
        elif self.normalize_coords == "min":
            min_dim = min(H, W, D)
            coords_h = torch.arange(0.5, H, **dd) / min_dim
            coords_w = torch.arange(0.5, W, **dd) / min_dim
            coords_d = torch.arange(0.5, D, **dd) / min_dim
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
            coords_d = torch.arange(0.5, D, **dd) / D
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij"),
            dim=-1
        )  # [H, W, D, 3]
        coords = coords.flatten(0, 2)                  # [HWD, 3]
        coords = 2.0 * coords - 1.0                    # [-1, +1]

        # Optional train-time augmentations on coords (DINOv3)
        if self.training and self.shift_coords is not None:
            shift_hwd = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift_hwd[None, :]

        if self.training and self.jitter_coords is not None:
            jit_max = np.log(self.jitter_coords); jit_min = -jit_max
            jitter = torch.empty(3, **dd).uniform_(jit_min, jit_max).exp()
            coords = coords * jitter[None, :]

        if self.training and self.rescale_coords is not None:
            r_max = np.log(self.rescale_coords); r_min = -r_max
            rescale = torch.empty(1, **dd).uniform_(r_min, r_max).exp()
            coords = coords * rescale

        # --- Build angles per axis, then concatenate across axes ---
        # coords: [N, 3] ; periods: [head_dim // 6]
        # angles: [N, 3, head_dim // 6] -> flatten(1, 2) -> [N, head_dim // 2] -> tile(2) -> [N, head_dim]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [N, 3, head_dim // 6]
        angles = angles.flatten(1, 2)                                            # [N, head_dim // 2]
        angles = angles.tile(2)                                                  # [N, head_dim]

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos
