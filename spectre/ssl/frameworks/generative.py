"""
Implementation of the MAE framework for Self-Supervised Learning (SSL).

This module provides the necessary components to train the MAE framework an is based on the 
original paper: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021), 
https://arxiv.org/abs/2111.06377
"""
import torch.nn as nn

from spectre.models import VisionTransformer
from spectre.ssl.models import MaskedVisionTransformer, MAEDecoder
from spectre.utils.models import (
    repeat_token, 
    set_at_index, 
    get_at_index, 
    random_token_mask, 
    patchify,
)


class MAE(nn.Module):
    def __init__(
            self,
            backbone: VisionTransformer,
            mask_ratio: float = 0.75,
            decoder_dim: int = 512,
            decoder_depth: int = 8,
            decoder_num_heads: int = 16,
            mlp_ratio: float = 4.0,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
        ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = backbone.patch_embed.patch_size

        self.backbone = MaskedVisionTransformer(
            vit=backbone,
        )
        self.backbone.mask_token = None  # remove ununsed mask token from encoder
        self.sequence_length = self.backbone.sequence_length

        self.decoder = MAEDecoder(
            num_patches=backbone.patch_embed.num_patches,
            patch_size=self.patch_size,
            in_chans=backbone.patch_embed.proj.in_channels,
            embed_dim=backbone.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = get_at_index(patches, idx_mask - 1)
        return x_pred, target
