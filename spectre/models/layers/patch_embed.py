import math
from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectre.utils.utils import to_3tuple
from spectre.utils.models import resample_patch_embed


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = (128, 128, 64),
            patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches
    
    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_3tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv3d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
    
    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size
        
    def dynamic_feat_size(self, img_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return (
                math.ceil(img_size[0] / self.patch_size[0]), 
                math.ceil(img_size[1] / self.patch_size[1]),
                math.ceil(img_size[2] / self.patch_size[2]),
            )
        else:
            return (
                img_size[0] // self.patch_size[0], 
                img_size[1] // self.patch_size[1],
                img_size[2] // self.patch_size[2],
            )

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
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)