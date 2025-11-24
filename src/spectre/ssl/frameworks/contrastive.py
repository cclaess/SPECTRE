"""
Implementation of the DINO framework for Self-Supervised Learning (SSL).

This module provides the necessary components to train the DINO framework an is based on the 
original paper: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021), 
https://arxiv.org/abs/2104.14294
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from spectre.ssl.models import MaskedVisionTransformer
from spectre.ssl.heads import DINOProjectionHead
from spectre.utils import (
    deactivate_requires_grad_and_to_eval, 
    update_drop_path_rate,
)


class DINO(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        freeze_last_layer: int = -1,
    ):
        super().__init__()

        self.backbone_student = backbone
        self.head_student = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=freeze_last_layer,
        )

        self.backbone_teacher = deepcopy(backbone)
        self.head_teacher = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad_and_to_eval(self.backbone_teacher)
        deactivate_requires_grad_and_to_eval(self.head_teacher)

    @torch.no_grad()
    def forward_teacher(
        self, 
        global_views: torch.Tensor
    ) -> torch.Tensor:
        
        # global views
        teacher_global_cls_token = self.backbone_teacher(global_views).flatten(start_dim=1)
        teacher_global_cls_out = self.head_teacher(teacher_global_cls_token)
        
        return teacher_global_cls_out

    def forward_student(
        self,
        global_views: torch.Tensor,
        local_views: torch.Tensor,
    ) -> torch.Tensor:
        
        # global views
        student_global_cls_token = self.backbone_student(global_views).flatten(start_dim=1)
        student_global_cls_out = self.head_student(student_global_cls_token)

        # local views
        student_local_cls_token = self.backbone_student(local_views).flatten(start_dim=1)
        student_local_cls_out = self.head_student(student_local_cls_token)

        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)

        return student_cls_out
    
    def forward(
        self, 
        global_views: torch.Tensor,
        local_views: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        teacher_out = self.forward_teacher(global_views)
        student_out = self.forward_student(global_views, local_views)

        return teacher_out, student_out


class DINOv2(nn.Module):
    def __init__(
        self, 
        backbone: "VisionTransformer", 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        ibot_seperate_head: bool = False,
        student_drop_path_rate: float = 0.1,
        freeze_last_layer: int = -1,
    ):
        super().__init__()

        self.backbone_student = MaskedVisionTransformer(vit=backbone)
        update_drop_path_rate(
            self.backbone_student.vit, 
            drop_path_rate=student_drop_path_rate
        )
        self.head_student_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, 
            freeze_last_layer=freeze_last_layer,
        )
        if ibot_seperate_head:
            self.head_student_ibot = DINOProjectionHead(
                input_dim, hidden_dim, bottleneck_dim, output_dim,
                freeze_last_layer=freeze_last_layer,
            )
        else:
            self.head_student_ibot = self.head_student_dino

        self.backbone_teacher = deepcopy(self.backbone_student)
        deactivate_requires_grad_and_to_eval(self.backbone_teacher)
        self.head_teacher_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad_and_to_eval(self.head_teacher_dino)
        if ibot_seperate_head:
            self.head_teacher_ibot = DINOProjectionHead(
                input_dim, hidden_dim, bottleneck_dim, output_dim,
            )
            deactivate_requires_grad_and_to_eval(self.head_teacher_ibot)
        else:
            self.head_teacher_ibot = self.head_teacher_dino
    
    @torch.no_grad()
    def forward_teacher(
        self, 
        global_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        teacher_features = self.backbone_teacher.encode(global_views, mask=None)
        teacher_global_cls_token = teacher_features[:, 0]

        teacher_global_cls_out = self.head_teacher_dino(teacher_global_cls_token)
        teacher_global_masked_out = self.head_teacher_ibot(teacher_features[mask])

        return teacher_global_cls_out, teacher_global_masked_out

    def forward_student(
        self, 
        global_views: torch.Tensor, 
        local_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        
        # global views
        student_features = self.backbone_student.encode(global_views, mask=mask)
        student_global_cls_token = student_features[:, 0]
        student_global_masked_features = student_features[mask]

        student_global_cls_out = self.head_student_dino(student_global_cls_token)
        student_global_masked_out = self.head_student_ibot(student_global_masked_features)

        # local views
        student_local_cls_token = self.backbone_student.encode(local_views, mask=None)[:, 0]
        student_local_cls_out = self.head_student_dino(student_local_cls_token)

        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)

        return student_cls_out, student_global_masked_out

    def forward(
        self, 
        global_views: torch.Tensor,
        local_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        teacher_cls_out, teacher_masked_out = self.forward_teacher(global_views, mask)
        student_cls_out, student_masked_out = self.forward_student(global_views, local_views, mask)

        return teacher_cls_out, teacher_masked_out, student_cls_out, student_masked_out
