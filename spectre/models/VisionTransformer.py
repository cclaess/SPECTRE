import math
from typing import Tuple, Union, Optional, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as VisionTransformerTimm

from spectre.ssl.utils import to_3tuple


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = 224,
            patch_size: Union[int, Tuple[int, int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if img_size is not None:
            self.img_size = to_3tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = math.prod(self.grid_size)
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                assert D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]})."
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[0] == 0, \
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert W % self.patch_size[1] == 0, \
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                assert D % self.patch_size[2] == 0, \
                    f"Input depth ({D}) should be divisible by patch size ({self.patch_size[2]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        x = self.norm(x)
        return x


class VisionTransformer(VisionTransformerTimm):
    def __init__(
            self, 
            img_size: Union[int, Tuple[int, int, int]] = 224,
            patch_size: Union[int, Tuple[int, int, int]] = 16,
            embed_layer: Callable = PatchEmbed,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_layer=embed_layer, 
            *args, 
            **kwargs
        )
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], *grid_size, -1).permute(0, 4, 1, 2, 3).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


def vit_tiny_patch16_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )

def vit_small_patch16_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )

def vit_base_patch16_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )

def vit_base_patch32_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(32, 32, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )
