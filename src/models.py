"""模型定义：基于 ImageNet 预训练 ResNet50 的迁移学习模型。

分类头统一替换 ResNet50 原始 fc 层（2048 -> num_classes），这是
torchvision 迁移学习的标准做法。参赛交付版本用的是「保留原 1000 类
fc 再叠加一层新线性头」的结构，重构时已规范化；如需加载当年的
.pth 权重或复现交付 ONNX，传 ``legacy_head=True`` 构建旧结构即可。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .classes import NUM_CLASSES

# ImageNet 归一化参数（torchvision 官方预训练模型统一使用）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """优先返回 CUDA 设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resnet50(pretrained: bool) -> nn.Module:
    """构建 ResNet50 骨干，兼容新旧版 torchvision 的权重参数。"""
    if pretrained:
        try:
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:  # 旧版 torchvision 无 weights 参数
            return models.resnet50(pretrained=True)
    return models.resnet50()


def build_disease_classifier(pretrained: bool = True, legacy_head: bool = False) -> nn.Module:
    """构建 24 类皮肤病分类模型。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重（加载已有权重微调时传 False）。
        legacy_head: True 时构建参赛交付版本的旧结构（保留原 1000 类 fc，
            叠加 Linear(1000, 24)），仅用于加载历史权重；新训练用默认结构。
    """
    model = _resnet50(pretrained)
    if legacy_head:
        model.add_module("add_linear", nn.Linear(1000, NUM_CLASSES))
    else:
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_body_verify_model(pretrained: bool = True, legacy_head: bool = False) -> nn.Module:
    """构建人体校验模型（人体 / 非人体二分类），APP 两级推理的第一级。

    legacy_head=True 时为参赛版本的旧结构：单输出打分头
    Linear(1000, 1)，对单输出做 softmax 恒返回类别 0，推理侧实际
    无法拦截图片（移动端按原始分数做阈值判断）。重构版默认为
    标准二分类头，两级过滤真正生效。
    """
    model = _resnet50(pretrained)
    if legacy_head:
        model.add_module("add_linear", nn.Linear(1000, 1))
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model
