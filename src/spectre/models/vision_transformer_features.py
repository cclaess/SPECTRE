import os
import math
from functools import partial
from urllib.parse import urlparse
from typing import Union, Callable, Literal, Optional, Type, Set, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from timm.layers import PatchDropout, AttentionPoolLatent
from huggingface_hub import hf_hub_download, load_state_dict_from_file

from spectre.utils import  global_pool_nlc, to_3tuple, resample_abs_pos_embed
from spectre.models.vision_transformer import Block
from spectre.models.layers import RotaryPositionEmbedding


class FeatureVisionTransformer(nn.Module):
    """ Vision Transformer that accepts flattened patches as input.
    """
    def __init__(
            self, 
            grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            patch_dim: int = 768,
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
            dynamic_grid_size: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Optional[Union[Callable, Type[torch.nn.Module]]] = None,
            act_layer: Optional[Union[Callable, Type[torch.nn.Module]]] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            num_patches: Number of patches in the input.
            patch_dim: Dimension of each flattened input patch.
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
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
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
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn', 'rope')
        assert attn_mode in ('mha', 'mqa', 'mla')
        assert grid_size is not None or pos_embed in ('', 'none', 'rope')
        rope_kwargs = {} if rope_kwargs is None else dict(rope_kwargs)
        rope_kwargs.setdefault("dtype", torch.float32)  # robust with mixed-precision
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.grid_size = None if grid_size is None else to_3tuple(grid_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_grid_size = dynamic_grid_size or pos_embed == 'rope'

        self.num_patches = None if grid_size is None else int(math.prod(grid_size))
        self.patch_proj = nn.Linear(patch_dim, embed_dim, proj_bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        self.pos_embed, self.rope, self.requires_per_sample_rope = None, None, False
        if pos_embed == 'learn':
            embed_len = self.num_patches if no_embed_class else self.num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        if pos_embed == 'rope':
            self.rope = RotaryPositionEmbedding(
                embed_dim=embed_dim,
                num_heads=num_heads,
                **rope_kwargs,
            )
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
            dict(module=f'blocks.{i}', num_chs=embed_dim) for i in range(depth)]
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

    def _pos_embed(
        self, 
        x: torch.Tensor,
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        if self.pos_embed is None and self.rope is None:
            return x.view(x.shape[0], -1, x.shape[-1])
        
        assert grid_size is not None or not self.dynamic_grid_size
        pos_embed, rope = None, None
        if self.pos_embed is not None:
            if self.dynamic_grid_size:
                H, W, D = to_3tuple(grid_size)
                prev_grid_size = self.grid_size
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed, 
                    new_size=(H, W, D),
                    old_size=prev_grid_size,
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
            else:
                pos_embed = self.pos_embed

        if self.rope is not None:
            B = x.shape[0]            
            H, W, D = to_3tuple(grid_size)
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


    def forward_features(
        self, 
        x: torch.Tensor, 
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        assert x.ndim == 3, f"Expected input with 3 dimensions (B, N, C), got {x.ndim}."

        x = self.patch_proj(x)
        x, rope = self._pos_embed(x, grid_size)
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

    def forward(
        self, 
        x: torch.Tensor, 
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        x = self.forward_features(x, grid_size)
        x = self.forward_head(x)
        return x
    
    @classmethod
    def from_pretrained(
            cls,
            checkpoint_path_or_url: Union[str, os.PathLike],
            verbose: bool = True,
            **kwargs
    ) -> 'FeatureVisionTransformer':
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


def feat_vit_tiny(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Tiny model.
    """
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=192,
        depth=2,
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs)


def feat_vit_small(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Small model.
    """
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=384,
        depth=2,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs)


def feat_vit_base(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Base model.
    """
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=768,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs)


def feat_vit_large(
    patch_dim,
    checkpoint_path_or_url: Optional[str] = None,
    **kwargs,
) -> FeatureVisionTransformer:
    """Feature ViT-Large model.
    """
    kwargs = dict(
        patch_dim=patch_dim,
        embed_dim=1080,
        depth=4,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if checkpoint_path_or_url is not None:
        return FeatureVisionTransformer.from_pretrained(checkpoint_path_or_url, **kwargs)
    return FeatureVisionTransformer(**kwargs)
