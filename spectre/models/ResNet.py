import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, create_aa, create_attn, get_act_layer, get_norm_layer
from timm.models.resnet import (
    get_padding,
    BasicBlock as BasicBlockTimm,
    Bottleneck as BottleneckTimm,
    ResNet as ResNetTimm,
)


class BasicBlock(BasicBlockTimm):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm3d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super().__init__(
            inplanes, planes, stride, downsample, cardinality, base_width, reduce_first, 
            dilation, first_dilation, act_layer, norm_layer, attn_layer, aa_layer, drop_block, drop_path
        )

        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv3d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.conv2 = nn.Conv3d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)


class Bottleneck(BottleneckTimm):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm3d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super().__init__(
            inplanes, planes, stride, downsample, cardinality, base_width, reduce_first,
            dilation, first_dilation, act_layer, norm_layer, attn_layer, aa_layer, drop_block, drop_path
        )

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv3d(inplanes, first_planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv3d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.conv3 = nn.Conv3d(width, outplanes, kernel_size=1, bias=False)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        layers: Tuple[int, ...],
        num_classes: int = 1000,
        in_chans: int = 1,
        stem_width: int = 64,
        channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm3d,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = act_layer
        norm_layer = norm_layer
        
        self.conv1 = nn.Conv3d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(stem_width)
        self.act1 = act_layer(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, norm_layer=norm_layer)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(planes, planes, stride, downsample, norm_layer=norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def resnet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], *args, **kwargs)

def resnet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], *args, **kwargs)

def resnet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)
