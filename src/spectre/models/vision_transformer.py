import os
from functools import partial
from urllib.parse import urlparse
from typing import (
    Tuple, Union, Callable, Sequence, 
    Literal, Optional, Type, Set, List,
)

import torch
import torch.nn as nn
from timm.layers import PatchDropout, AttentionPoolLatent
from timm.models.vision_transformer import LayerScale, DropPath, Mlp
from huggingface_hub import hf_hub_download, load_state_dict_from_file

from spectre.models.layers import (
    PatchEmbed, 
    Attention, 
    RotaryPositionEmbedding,
)
from spectre.utils import (
    resample_abs_pos_embed, 
    feature_take_indices, 
    global_pool_nlc,
)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_mode: str = 'mha',
        q_proj_dim: Optional[int] = None,
        kv_proj_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            mode=attn_mode,
            q_proj_dim=q_proj_dim,
            kv_proj_dim=kv_proj_dim,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        rope = None
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope=rope)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with 3D Patch Embedding
    """
    def __init__(
        self, 
        img_size: Union[int, Tuple[int, int, int]] = (128, 128, 64),
        patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
        in_chans: int = 1,
        num_classes: int = 1000,
        global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        attn_mode: str = 'mha',
        q_proj_dim: Optional[int] = None,
        kv_proj_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = 'learn',
        no_embed_class: bool = False,
        rope_kwargs: Optional[dict] = None,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[Union[Callable, Type[torch.nn.Module]]] = None,
        norm_layer: Optional[Union[Callable, Type[torch.nn.Module]]] = None,
        act_layer: Optional[Union[Callable, Type[torch.nn.Module]]] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            attn_mode: Attention mode ('mha', 'mqa', 'mla').
            q_proj_dim: Query projection dimension for 'mla' mode.
            kv_proj_dim: Key, value projection dimension for 'mla' mode.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            pos_embed: Type of position embedding to use (default: 'learn').
            no_embed_class: Don't include position embeddings for class (or reg) tokens for learnable pos_embed.
            rope_kwargs: Additional arguments for rotary position embedding.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn', 'rope')
        assert attn_mode in ('mha', 'mqa', 'mla')
        rope_kwargs = {} if rope_kwargs is None else dict(rope_kwargs)
        rope_kwargs.setdefault("dtype", torch.float32)  # robust with mixed-precision
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = embed_norm_layer
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size or pos_embed == 'rope'

        embed_args = {}
        if self.dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWDC"))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed, self.rope, self.requires_per_sample_rope = None, None, False
        if pos_embed == 'learn':
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        if pos_embed == 'rope':
            self.rope = RotaryPositionEmbedding(
                embed_dim=embed_dim,
                num_heads=num_heads,
                **rope_kwargs,
            )
            self.requires_per_sample_rope = any([
                self.rope.shift_coords is not None,
                self.rope.jitter_coords is not None,
                self.rope.rescale_coords is not None,
            ])
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                attn_mode=attn_mode,
                q_proj_dim=q_proj_dim,
                kv_proj_dim=kv_proj_dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int, int]] = None,
            patch_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Method updates the input image resolution, patch size

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size: New patch size, if None existing patch size is used
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=self.patch_embed.grid_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    verbose=True,
                ))

    def _pos_embed(self, x: torch.Tensor):
        if self.pos_embed is None and self.rope is None:
            return x.view(x.shape[0], -1, x.shape[-1]), None
        
        B, H, W, D, C = x.shape
        x = x.view(B, -1, C)
        pos_embed, rope = None, None
        if self.pos_embed is not None:
            if self.dynamic_img_size:
                prev_grid_size = self.patch_embed.grid_size
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=(H, W, D),
                    old_size=prev_grid_size,
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
            else:
                pos_embed = self.pos_embed
        
        if self.rope is not None:
            if self.requires_per_sample_rope:
                rope = [self.rope(H=H, W=W, D=D) for _ in range(B)]
            else:
                rope = self.rope(H=H, W=W, D=D)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed

        return self.pos_drop(x), rope
        

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHWD',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHWD', 'NLC'), 'Output format must be one of NCHWD or NLC.'
        reshape = output_fmt == 'NCHWD'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width, depth = x.shape
        x = self.patch_embed(x)
        x, rope = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x, rope=rope)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            H, W, D = self.patch_embed.dynamic_feat_size((height, width, depth))
            intermediates = [y.reshape(B, H, W, D -1).permute(0, 4, 1, 2, 3).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
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
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x, rope = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x, rope=rope)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    @classmethod
    def from_pretrained(
            cls,
            checkpoint_path_or_url: Union[str, os.PathLike],
            verbose: bool = True,
            **kwargs
    ) -> 'VisionTransformer':
        """Load pretrained model weights from a local path or a URL."""
        model = cls(**kwargs)

        def _is_url(path: str) -> bool:
            try:
                parsed = urlparse(str(path))
                return parsed.scheme in ('http', 'https')
            except Exception:
                return False
            
        def _is_hf_url(path: str) -> bool:
            try:
                parsed = urlparse(str(path))
                return 'huggingface.co' in parsed.netloc
            except Exception:
                return False

        if _is_hf_url(checkpoint_path_or_url):
            if verbose:
                print(f"Downloading pretrained weights from Hugging Face URL: {checkpoint_path_or_url}")
            # Extract repo_id and filename from the URL
            parsed = urlparse(checkpoint_path_or_url)
            parts = parsed.path.strip('/').split('/')
            repo_id = '/'.join(parts[:2])  # e.g., 'cclaess/SPECTRE'
            filename = parts[-1]           # e.g., 'spectre_backbone_vit_large_patch16_128.pt'

            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            state_dict = load_state_dict_from_file(local_path, map_location='cpu')
        elif _is_url(checkpoint_path_or_url):
            if verbose:
                print(f"Downloading pretrained weights from URL: {checkpoint_path_or_url}")
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint_path_or_url, map_location='cpu', weights_only=False, progress=verbose)
        else:
            local_path = os.fspath(checkpoint_path_or_url)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Checkpoint file not found: {local_path}")
        if verbose:
            print(f"Loading checkpoint from local path: {local_path}")
            state_dict = torch.load(local_path, map_location='cpu', weights_only=False)

        msg = model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"Loaded pretrained weights with msg: {msg}")
        return model


def vit_tiny_patch16_128(
    checkpoint_path_or_url: Optional[str] = None, 
    **kwargs
) -> VisionTransformer:
    """ViT-Tiny model with 3D patch embedding, patch size [16, 16, 8] and input size [128, 128, 64].
    """
    kwargs = dict(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=192,
        depth=12,
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return VisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return VisionTransformer(**kwargs)


def vit_small_patch16_128(
    checkpoint_path_or_url: Optional[str] = None, 
    **kwargs
) -> VisionTransformer:
    """ViT-Small model with 3D patch embedding, patch size [16, 16, 8] and input size [128, 128, 64].
    """
    kwargs = dict(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=384,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return VisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return VisionTransformer(**kwargs)


def vit_base_patch16_128(
    checkpoint_path_or_url: Optional[str] = None, 
    **kwargs
) -> VisionTransformer:
    """ViT-Base model with 3D patch embedding, patch size [16, 16, 8] and input size [128, 128, 64].
    """
    kwargs = dict(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return VisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return VisionTransformer(**kwargs)
    

def vit_base_patch32_128(
    checkpoint_path_or_url: Optional[str] = None, 
    **kwargs
) -> VisionTransformer:
    """ViT-Base model with 3D patch embedding, patch size [32, 32, 16] and input size [128, 128, 64].
    """
    kwargs = dict(
        img_size=(128, 128, 64),
        patch_size=(32, 32, 16),
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return VisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return VisionTransformer(**kwargs)


def vit_large_patch16_128(
    checkpoint_path_or_url: Optional[str] = None, 
    **kwargs
) -> VisionTransformer:
    """ViT-Large model with 3D patch embedding, patch size [16, 16, 8] and input size [128, 128, 64].
    """
    kwargs = dict(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=1080,
        depth=24,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return VisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return VisionTransformer(**kwargs)


def vit_large_patch32_128(
    checkpoint_path_or_url: Optional[str] = None, 
    **kwargs
) -> VisionTransformer:
    """ViT-Large model with 3D patch embedding, patch size [32, 32, 16] and input size [128, 128, 64].
    """
    kwargs = dict(
        img_size=(128, 128, 64),
        patch_size=(32, 32, 16),
        embed_dim=1080,
        depth=24,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return VisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return VisionTransformer(**kwargs)
