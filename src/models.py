# -*- coding: utf-8 -*-
"""模型构建。

说明：项目交付的权重与安卓端 ONNX 模型均基于以下结构训练——
保留 ImageNet 预训练 ResNet50 的原始 1000 类 fc 层，
在其后叠加新的分类头 ``add_linear``（Linear(1000, num_classes)）。
该结构略显非常规，但为保证与已有模型权重可复现、可加载，
这里保持原始结构不变。
"""

try:  # 兼容「作为 src 包导入」与「在 src/ 目录内直接运行」两种方式
    from .classes import NUM_CLASSES
except ImportError:  # pragma: no cover
    from classes import NUM_CLASSES

import torch
from torch import nn
from torchvision import models


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """构建与交付模型结构一致的 ResNet50（原 fc 层 + add_linear 新头）。"""
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:  # 旧版 torchvision 无 weights 参数
        model = models.resnet50(pretrained=pretrained)
    model.add_module("add_linear", nn.Linear(1000, num_classes))
    return model


def build_disease_classifier(pretrained: bool = True) -> nn.Module:
    """24 类皮肤病分类模型。"""
    return build_resnet50(NUM_CLASSES, pretrained)


def build_body_verify_model(pretrained: bool = True) -> nn.Module:
    """人体校验模型：单输出打分头（结构与交付权重 / ONNX 一致）。"""
    return build_resnet50(1, pretrained)
