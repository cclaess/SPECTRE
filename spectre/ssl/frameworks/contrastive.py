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
from spectre.ssl.heads import DINOProjectionHead, DINOv2ProjectionHead
from spectre.utils import deactivate_requires_grad, update_drop_path_rate


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
        ibot_seperate_head: bool = False,
        student_drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.student_backbone = MaskedVisionTransformer(vit=backbone)
        update_drop_path_rate(
            self.student_backbone.vit, 
            drop_path_rate=student_drop_path_rate
        )
        self.student_head_dino = DINOv2ProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        if ibot_seperate_head:
            self.student_head_ibot = DINOv2ProjectionHead(
                input_dim, hidden_dim, bottleneck_dim, output_dim,
            )
        else:
            self.student_head_ibot = self.student_head_dino

        self.teacher_backbone = deepcopy(self.student_backbone)
        deactivate_requires_grad(self.teacher_backbone)
        self.teacher_head_dino = DINOv2ProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad(self.teacher_head_dino)
        if ibot_seperate_head:
            self.teacher_head_ibot = DINOv2ProjectionHead(
                input_dim, hidden_dim, bottleneck_dim, output_dim,
            )
            deactivate_requires_grad(self.teacher_head_ibot)
        else:
            self.teacher_head_ibot = self.teacher_head_dino

    def forward_student(
        self, 
        global_views: torch.Tensor, 
        local_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        
        # global views
        student_features = self.student_backbone.encode(global_views, mask=mask)
        student_global_cls_token = student_features[:, 0]
        student_global_masked_features = student_features[mask]

        student_global_cls_out = self.student_head_dino(student_global_cls_token)
        student_global_masked_out = self.student_head_ibot(student_global_masked_features)

        # local views
        student_local_cls_token = self.student_backbone.encode(local_views, mask=None)[:, 0]
        student_local_cls_out = self.student_head_dino(student_local_cls_token)

        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)

        return student_cls_out, student_global_masked_out
    
    def forward_teacher(
            self, 
            global_views: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:

        teacher_features = self.teacher_backbone.encode(global_views, mask=None)
        teacher_global_cls_token = teacher_features[:, 0]

        teacher_global_cls_out = self.teacher_head_dino(teacher_global_cls_token)
        teacher_global_masked_out = self.teacher_head_ibot(teacher_features[mask])

        return teacher_global_cls_out, teacher_global_masked_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_backbone(x)
