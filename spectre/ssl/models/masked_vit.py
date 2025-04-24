import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from spectre.models import VisionTransformer
from spectre.utils.models import (
    mask_bool,
    get_at_index, 
    mask_at_index,
    resample_abs_pos_embed,
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
        vit: VisionTransformer,
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
        tokens: torch.Tensor = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)
        # apply Transformer blocks
        tokens = self.vit.blocks(tokens)
        # normalize
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
                Tensor with shape (batch_size, channels, image_height, image_width).
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

        # convert images to tokens
        tokens = self.images_to_tokens(images)
        # add prefix tokens if needed
        tokens = self.prepend_prefix_tokens(tokens)

        if idx_mask is not None:
            tokens = mask_at_index(
                tokens=tokens, index=idx_mask, mask_token=self.mask_token
            )
        elif mask is not None:
            tokens = mask_bool(
                tokens=tokens, mask=mask, mask_token=self.mask_token
            )

        # add positional encoding
        tokens = self.add_pos_embed(tokens)

        if idx_keep is not None:
            tokens = get_at_index(tokens, idx_keep)

        return tokens

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

    def add_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional embeddings to the input tensor based on the Vision Transformer
        (ViT) architecture in vit.

        Args:
            x:
                Input tensor with shape (batch_size, self.sequence_length, vit.embed_dim).

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """

        x_prefix = x[:, : self.vit.num_prefix_tokens, :]
        x = x[:, self.vit.num_prefix_tokens :, :]
        if self.vit.dynamic_img_size:
            x = x.transpose(1, 2)  # NLC -> NCL
            batch_size = x.size(0)
            num_channels = x.size(1)
            grid_size = round(math.pow(x.size(2), 1 / 3))
            grid_size = (grid_size, grid_size, grid_size)
            x = x.view(
                batch_size,
                num_channels,
                grid_size[0],
                grid_size[1],
                grid_size[2],
            )  # NCL -> NCHWD

            # NCHWD -> NHWDC
            x = x.permute(0, 2, 3, 4, 1)
            B, H, W, D, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.vit.pos_embed,
                (H, W, D),
                num_prefix_tokens=(
                    0 if self.vit.no_embed_class else self.vit.num_prefix_tokens
                ),
                old_size=self.vit.patch_embed.grid_size,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.vit.pos_embed

        if self.vit.no_embed_class:
            x = x + pos_embed
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
        else:
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
            x = x + pos_embed
        out: torch.Tensor = self.vit.pos_drop(x)
        return out

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

        # initialize_3d_sine_cosine_positional_embedding(
        #     pos_embedding=self.vit.pos_embed, has_class_token=self.vit.has_class_token
        # )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
