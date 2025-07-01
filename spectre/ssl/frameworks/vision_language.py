"""
Implementation of the CLIP framework for text-image feature alignment.

This module provides the necessary components to train the CLIP framework an is based on the 
original paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021), 
https://arxiv.org/abs/2103.00020

Addional resources:
Hamamci et al., "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography" (2024),
https://arxiv.org/abs/2403.17834
"""
from typing import Optional

import torch
import torch.nn as nn

from spectre.ssl.heads import SigLIPProjectionHead


class SigLIP(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        text_backbone: nn.Module,
        image_feature_comb: Optional[nn.Module] = None,
        image_embed_dim: int = 768,
        text_embed_dim: int = 1536,
        projection_dim: int = 512,
        backbone_is_class_token: bool = False,
        backbone_combine_features: bool = True,
        feature_comb_is_class_token: bool = False,
        feature_comb_combine_features: bool = True,
    ):
        super().__init__()
        assert not backbone_is_class_token or not backbone_combine_features
        assert not feature_comb_is_class_token or not feature_comb_combine_features
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.image_feature_comb = image_feature_comb
        self.backbone_is_class_token = backbone_is_class_token
        self.backbone_combine_features = backbone_combine_features
        self.feature_comb_is_class_token = feature_comb_is_class_token
        self.feature_comb_combine_features = feature_comb_combine_features

        self.image_projection = SigLIPProjectionHead(
            input_dim=image_embed_dim,
            output_dim=projection_dim,
            freeze_last_layer=5,
        )
        self.text_projection = SigLIPProjectionHead(
            input_dim=text_embed_dim,
            output_dim=projection_dim,
            freeze_last_layer=5,
        )

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
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
        image_embeddings = self.image_backbone(images)
        if self.backbone_is_class_token:
            image_embeddings = image_embeddings.view(B, N, -1)  # (batch, crops, embed_dim)
        else:
            image_embeddings = image_embeddings.view(B, N, image_embeddings.shape[1], -1)  # (batch, crops, patches, embed_dim)
            if self.backbone_combine_features:
                image_embeddings = torch.cat([
                    image_embeddings[:, :, 0, :],  # class token
                    image_embeddings[:, :, 1:, :].mean(dim=2)  # mean of patch tokens
                ], dim=2)  # (batch, crops, embed_dim)
                torch.mean
            else:
                image_embeddings = image_embeddings[:, :, 0, :]

        if self.image_feature_comb is not None:
            image_embeddings = self.image_feature_comb(image_embeddings) # (batch, embed_dim)
        else:
            image_embeddings = image_embeddings.max(dim=1).values
        if self.feature_comb_combine_features:
            image_embeddings = torch.cat([
                image_embeddings[:, 0, :],  # class token
                image_embeddings[:, 1:, :].mean(dim=1)  # mean of patch tokens
            ], dim=1)
        else:
            if not self.feature_comb_is_class_token:
                image_embeddings = image_embeddings[:, 0, :]
        image_embeddings = self.image_projection(image_embeddings) # (batch, embed_dim)

        # Compute text embeddings
        text_embeddings = self.text_backbone(input_ids=text_tokens, attention_mask=attention_mask)
        text_embeddings = self.text_projection(text_embeddings.pooler_output) # (batch, embed_dim)

        return image_embeddings, text_embeddings
