"""
Implementation of the CLIP framework for text-image feature alignment.

This module provides the necessary components to train the CLIP framework an is based on the 
original paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021), 
https://arxiv.org/abs/2103.00020

Addional resources:
Hamamci et al., "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography" (2024),
https://arxiv.org/abs/2403.17834
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from spectre.ssl.heads import SigLIPProjectionHead
from spectre.utils import last_token_pool


class SigLIP(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        text_backbone: nn.Module,
        image_feature_comb: Optional[nn.Module] = None,
        image_embed_dim: int = 768,
        text_embed_dim: int = 1536,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.backbone_image = image_backbone
        self.backbone_text = text_backbone
        self.feature_comb_image = image_feature_comb

        self.projection_image = SigLIPProjectionHead(
            input_dim=image_embed_dim,
            output_dim=projection_dim,
            freeze_last_layer=1,
        )
        self.projection_text = SigLIPProjectionHead(
            input_dim=text_embed_dim,
            output_dim=projection_dim,
            freeze_last_layer=1,
        )

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        image_grid_size: Optional[Tuple[int, int, int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            images: Tensor of shape (batch, crops, channel, height, width, depth)
            text_tokens: Tensor of shape (batch, sequence_length)
            attention_mask: Tensor of shape (batch, sequence_length)

        Returns:
            logits_per_image: Tensor (batch, batch) with similarity scores.
            logits_per_text: Tensor (batch, batch) with similarity scores.
        """
        # reshape input to be (batch x crops, 1,  height, width, depth)
        B, N, C, H, W, D = images.shape
        images = images.view(B*N, C, H, W, D)

        # Compute image embeddings
        image_embeddings = self.backbone_image(images)
        image_embeddings = image_embeddings.view(B, N, image_embeddings.shape[1], -1)
        assert image_embeddings.shape[2] > 1, "Backbone must return class token and patch tokens"

        image_embeddings = torch.cat([
            image_embeddings[:, :, 0, :],  # class token
            image_embeddings[:, :, 1:, :].mean(dim=2)  # mean of patch tokens
        ], dim=2)  # (batch, crops, embed_dim * 2)

        if self.feature_comb_image is not None:
            image_embeddings = self.feature_comb_image(image_embeddings, grid_size=image_grid_size)
            assert image_embeddings.shape[1] > 1, \
                "Feature combination module must return class token and patch tokens"

            image_embeddings = torch.cat([
                image_embeddings[:, 0, :],  # class token
                image_embeddings[:, 1:, :].mean(dim=1)  # mean of patch tokens
            ], dim=1)

        image_embeddings = self.projection_image(image_embeddings)

        # Compute text embeddings
        text_embeddings = self.backbone_text(input_ids=text_tokens, attention_mask=attention_mask)
        text_embeddings = self.projection_text(
            last_token_pool(text_embeddings.last_hidden_state, attention_mask)
        )

        return image_embeddings, text_embeddings
