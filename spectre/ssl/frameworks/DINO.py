"""
Implementation of the DINO framework for Self-Supervised Learning (SSL).

This module provides the necessary components to train the DINO framework an is based on the 
original paper: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021), 
https://arxiv.org/abs/2104.14294
"""
from copy import deepcopy

import torch
import torch.nn as nn

from spectre.ssl.heads import DINOProjectionHead
from spectre.utils.models import deactivate_requires_grad


class DINO(nn.Module):
    def __init__(
            self, 
            backbone: nn.Module, 
            input_dim: int,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
            output_dim: int = 65536,
        ):
        super().__init__()

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1,
        )

        self.teacher_backbone = deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.student_backbone(x, pre_logits=True).flatten(start_dim=1)
        x = self.student_head(x)
        return x
    
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        x = self.teacher_backbone(x, pre_logits=True).flatten(start_dim=1)
        x = self.teacher_head(x)
        return x
