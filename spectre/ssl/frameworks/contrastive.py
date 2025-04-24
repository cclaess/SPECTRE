"""
Implementation of the DINO framework for Self-Supervised Learning (SSL).

This module provides the necessary components to train the DINO framework an is based on the 
original paper: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021), 
https://arxiv.org/abs/2104.14294
"""
from copy import deepcopy

import torch
import torch.nn as nn

from spectre.models import VisionTransformer
from spectre.ssl.models import MaskedVisionTransformer
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
    
    def forward(
        self, 
        global_crops: torch.Tensor, 
        local_crops: torch.Tensor
    ) -> torch.Tensor:
        cls_tokens_global = self.student_backbone(global_crops).flatten(start_dim=1)
        cls_tokens_local = self.student_backbone(local_crops).flatten(start_dim=1)

        cls_tokens_global_after_head = self.student_head(cls_tokens_global)
        cls_tokens_local_after_head = self.student_head(cls_tokens_local)
        
        return cls_tokens_global_after_head, cls_tokens_local_after_head
    
    def forward_teacher(
        self, 
        global_crops: torch.Tensor
    ) -> torch.Tensor:
        cls_tokens = self.teacher_backbone(global_crops).flatten(start_dim=1)
        cls_tokens_after_head = self.teacher_head(cls_tokens)
        return cls_tokens_after_head


class DINOv2(nn.Module):
    def __init__(
        self, 
        backbone: VisionTransformer, 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
    ):
        super().__init__()

        self.student_backbone = MaskedVisionTransformer(vit=backbone)
        self.student_head_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1,
        )
        self.student_head_ibot = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1,
        )

        self.teacher_backbone = deepcopy(backbone)
        self.teacher_head_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        self.teacher_head_ibot = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head_dino)
        deactivate_requires_grad(self.teacher_head_ibot)
    
    def forward(
        self, 
        global_crops: torch.Tensor, 
        local_crops: torch.Tensor,
        masks: torch.Tensor,
        mask_indices: list, 
        upperbound: int,
    ) -> torch.Tensor:
        x_global = self.student_backbone.encode(global_crops, mask=masks)
        x_local = self.student_backbone.encode(local_crops)

        cls_tokens_global = x_global[:, 0]
        patch_tokens_global = x_global[:, 1:]
        cls_tokens_local = x_local[:, 0]

        buffer_tensor = patch_tokens_global.new_zeros(
            upperbound, patch_tokens_global.shape[-1])
        buffer_tensor[:mask_indices.shape[0]].copy_(torch.index_select(
            patch_tokens_global.flatten(0, 1),
            dim=0,
            index=mask_indices,
        ))

        cls_tokens_global_after_head = self.student_head_dino(cls_tokens_global)
        patch_tokens_global_after_head = self.student_head_ibot(buffer_tensor)[
            :mask_indices.shape[0]
        ]
        cls_tokens_local_after_head = self.student_head_dino(cls_tokens_local)
        
        return cls_tokens_global_after_head, patch_tokens_global_after_head, cls_tokens_local_after_head
    
    def forward_teacher(
            self, 
            global_crops: torch.Tensor, 
            mask_indices: list, 
            upperbound: int,
        ) -> torch.Tensor:
        x = self.teacher_backbone.forward_features(global_crops)
        cls_tokens = x[:, 0]
        patch_tokens = x[:, 1:]

        buffer_tensor = patch_tokens.new_zeros(
            upperbound, patch_tokens.shape[-1])
        torch.index_select(
            patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices,
            out=buffer_tensor[:mask_indices.shape[0]],
        )

        cls_tokens_after_head = self.teacher_head_dino(cls_tokens)
        patch_tokens_after_head = self.teacher_head_ibot(buffer_tensor)[
            :mask_indices.shape[0]
        ]
           
        return cls_tokens_after_head, patch_tokens_after_head
