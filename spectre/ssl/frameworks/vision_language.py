"""
Implementation of the CLIP framework for text-image feature alignment.

This module provides the necessary components to train the CLIP framework an is based on the 
original paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021), 
https://arxiv.org/abs/2103.00020

Addional resources:
Hamamci et al., "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography" (2024),
https://arxiv.org/abs/2403.17834
"""
import os
from typing import Optional

import torch
import torch.nn as nn


class SigLIP(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        text_backbone: nn.Module,
        image_feature_comb: Optional[nn.Module] = None,
        image_embed_dim: int = 768,
        text_embed_dim: int = 1536,
        projection_dim: int = 768,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.image_feature_comb = image_feature_comb

        self.image_projection = nn.Linear(image_embed_dim, projection_dim)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim)

        nn.init.trunc_normal_(self.image_projection.weight, std=0.01)
        nn.init.trunc_normal_(self.text_projection.weight, std=0.01)

        # Initialize the bias to zero
        if self.image_projection.bias is not None:
            nn.init.zeros_(self.image_projection.bias)
        if self.text_projection.bias is not None:
            nn.init.zeros_(self.text_projection.bias)

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
        if self.image_feature_comb is not None:
            image_embeddings = self.image_feature_comb(image_embeddings.view(B, N, -1)) # (batch, embed_dim)
        image_embeddings = self.image_projection(image_embeddings) # (batch, embed_dim)

        # Compute text embeddings
        text_embeddings = self.text_backbone(input_ids=text_tokens, attention_mask=attention_mask)
        text_embeddings = self.text_projection(text_embeddings.pooler_output) # (batch, embed_dim)

        return image_embeddings, text_embeddings
