from typing import Type, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import use_fused_attn


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mode: str = "mha",
            q_proj_dim: Optional[int] = None,
            kv_proj_dim: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.mode = mode.lower()
        assert self.mode in ["mha", "mqa", "mla"], "Attention mode must be 'mha', 'mqa', or 'mla'"
        assert not (self.mode == "mla" and kv_proj_dim is None), "kv_proj_dim must be provided for 'mla' mode"
        assert not (self.mode == "mla" and q_proj_dim is None), "q_proj_dim must be provided for 'mla' mode"
        
        if self.mode == "mha":
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # Key and value pair for every head
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        elif self.mode == "mqa":
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * self.head_dim, bias=qkv_bias)  # Key and value pair shared across heads
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        elif self.mode == "mla":
            self.q_proj = nn.Linear(dim, q_proj_dim, bias=qkv_bias)  # Projected query for every head
            self.kv_proj = nn.Linear(dim, kv_proj_dim, bias=qkv_bias)  # Projected key and value pair for every head
            self.q_norm = norm_layer(q_proj_dim) if qk_norm else nn.Identity()
            self.kv_norm = norm_layer(kv_proj_dim) if qk_norm else nn.Identity()
            self.q = nn.Linear(q_proj_dim, dim, bias=qkv_bias)  # Query for every head
            self.kv = nn.Linear(kv_proj_dim, 2 * dim, bias=qkv_bias)  # Key and value pair for every head
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if self.mode == "mha":
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.mode == "mqa":
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, N, 2, 1, self.head_dim).permute(2, 0, 3, 1, 4)
            kv = kv.expand(-1, -1, self.num_heads, -1, -1)  # Expand to match num_heads
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.mode == "mla":
            q = self.q_proj(x)
            kv = self.kv_proj(x)
            q, kv = self.q_norm(q), self.kv_norm(kv)  # Normalization on projections
            q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
