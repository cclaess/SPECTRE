"""
Implementation of the CLIP framework for text-image feature alignment.

This module provides the necessary components to train the CLIP framework an is based on the 
original paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021), 
https://arxiv.org/abs/2103.00020

Addional resources:
Hamamci et al., "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography" (2024),
https://arxiv.org/abs/2403.17834
"""
import math
from typing import Tuple

import torch.nn as nn


class CLIP(nn.Module):
    def __init__(
            self,
            image_backbone: nn.Module,
            text_backbone: nn.Module,
            grid_size: Tuple[int, int, int],
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.grid_size = grid_size
    
    def forward(self, images, texts):
        image_features = self.image_backbone(images)

        # MONAI dataloaders will push chunks of the same CT volume to the batch dimension
        # Reorder batch so that chunks of the same CT volume are together
        B, C  = image_features.shape
        B_eff = B // math.prod(self.grid_size)
        image_features = image_features.view(B_eff, -1, C)

        # Max-pool over chunks of the same CT volume
        image_features = image_features.max(dim=1).values

        text_features = self.text_backbone(texts)
        return image_features, text_features