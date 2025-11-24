from __future__ import annotations

from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn

from spectre.utils import (
    mask_bool,
    get_at_index, 
    mask_at_index,
    resample_abs_pos_embed,
    to_3tuple,
)


class MaskedVisionTransformer(nn.Module):
    """Masked Vision Transformer.

    Attributes:
        vit: The VisionTransformer object.
        mask_token: The mask token.
        use_mask_token: Whether to use the mask token.
    """

    def __init__(
        self,
        vit: "VisionTransformer",
        mask_token: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        )

        self._initialize_weights()

    @property
    def sequence_length(self) -> int:
        seq_len: int = self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens
        return seq_len

    def forward(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size, image_size).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be passed to the
                encoder.

        Returns:
            Tensor with shape (batch_size, vit.embed_dim) containing the
            encoded class token for every image.

        """
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep)
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == "avg":
            x = x[:, self.vit.num_prefix_tokens :].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]  # class token
        return x
    
    def forward_intermediates(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
        norm: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # preprocess images, convert to tokens and add positional embeddings
        tokens = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)

        intermediates: List[torch.Tensor] = []
        for blk in self.vit.blocks:
            tokens = blk(tokens)
            intermediates.append(self.vit.norm(tokens) if norm else tokens)

        # normalize
        out: torch.Tensor = self.vit.norm(tokens)

        return out, intermediates

    def encode(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input images.

        Args:
            input:
                Batch of input images.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be masked with
                self.mask_token.
        Returns:
            Batch of encoded output tokens.
        """
        tokens, rope = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)

        # apply Transformer blocks
        for blk in self.vit.blocks:
            tokens = blk(tokens, rope=rope)
        tokens = self.vit.norm(tokens)
        return tokens

    def preprocess(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert images to tokens, add positional embeddings, and apply masking.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width, 
                image_depth).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be returned.
                Is applied after any masking operation.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be masked with
                self.mask_token.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            preprocessed tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned. Any class or prefix tokens are prepended to the
            sequence.
        """
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set at the same time.")
        
        _, _, H, W, D = images.shape

        # convert images to tokens
        tokens: torch.Tensor = self.images_to_tokens(images)
        # add prefix tokens if needed
        tokens: torch.Tensor = self.prepend_prefix_tokens(tokens)

        if idx_mask is not None:
            tokens = mask_at_index(
                tokens=tokens, index=idx_mask, mask_token=self.mask_token
            )
        elif mask is not None:
            tokens = mask_bool(
                tokens=tokens, mask=mask, mask_token=self.mask_token
            )

        # add positional encoding
        tokens, rope = self.add_pos_embed(tokens, img_size=(H, W, D))

        if idx_keep is not None:
            tokens = get_at_index(tokens, idx_keep)
            if rope is not None:
                if isinstance(rope, list):
                    rope = [get_at_index(r, idx_keep) for r in rope]
                else:
                    rope = get_at_index(rope, idx_keep)

        return tokens, rope

    def images_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, vit.patch_embed.num_patches, vit.embed_dim)
            containing the patch tokens (excluding prefix tokens).
        """
        tokens: torch.Tensor = self.vit.patch_embed(images)
        if self.vit.dynamic_img_size:
            tokens = tokens.permute(0, 4, 1, 2, 3)  # NHWDC -> NCHWD
            tokens = tokens.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        return tokens

    def prepend_prefix_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Adds prefix tokens to image patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, vit.patch_embed.num_patches, vit.embed_dim)
                containing the image patch tokens

        Returns:
            Tensor with shape (batch_size, self.sequence_length, vit.embed_dim) containing
            the image patch tokens and prefix tokens.
        """
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(x.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(
        self, x: torch.Tensor, img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] | None]:
        """Adds positional embeddings to the input tensor based on the Vision Transformer
        (ViT) architecture in vit.

        Args:
            x: Input tensor with shape (batch_size, self.sequence_length, vit.embed_dim).
            img_size: Image size as an integer or tuple (H, W, D). Only needed if
            self.vit.dynamic_img_size is True.

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
            Rotary positional embeddings (RoPE) if self.vit.rope is not None.
        """
        if self.vit.pos_embed is None and self.vit.rope is None:
            return x, None
        
        assert img_size is not None or not self.vit.dynamic_img_size
        
        if self.vit.pos_embed is not None:
            if self.vit.dynamic_img_size:
                H, W, D = to_3tuple(img_size)
                prev_grid_size = self.vit.patch_embed.grid_size
                new_size = self.vit.patch_embed.dynamic_feat_size((H, W, D))
                pos_embed = resample_abs_pos_embed(
                    self.vit.pos_embed,
                    new_size=new_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=(
                        0 if self.vit.no_embed_class else self.vit.num_prefix_tokens
                    )
                )
            else:
                pos_embed = self.vit.pos_embed
            
            if self.vit.no_embed_class:
                if self.vit.num_prefix_tokens:
                    prefix = x[:, : self.vit.num_prefix_tokens, :]
                    spatial = x[:, self.vit.num_prefix_tokens :, :]
                    spatial = spatial + pos_embed
                    x = torch.cat((prefix, spatial), dim=1)
                else:
                    x = x + pos_embed
            else:
                x = x + pos_embed

            # apply positional dropout (only for learned absolute pos)
            x = self.vit.pos_drop(x)

            return x, None

        else:  # rotary positional embedding
            B = x.shape[0]
            H, W, D = to_3tuple(img_size)

            feat_h, feat_w, feat_d = self.vit.patch_embed.dynamic_feat_size((H, W, D))
            if self.vit.requires_per_sample_rope:
                rope = [self.vit.rope(H=feat_h, W=feat_w, D=feat_d) for _ in range(B)]
            else:
                rope = self.vit.rope(H=feat_h, W=feat_w, D=feat_d)

            return x, rope

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.vit.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        if self.vit.has_class_token:
            nn.init.normal_(self.vit.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
