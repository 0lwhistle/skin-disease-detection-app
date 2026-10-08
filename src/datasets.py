"""数据集构建：ImageFolder 目录约定 + 训练/验证预处理策略。

数据目录结构（类别目录名即类别名，需与 ``classes.py`` 顺序一致）::

    data_dir/
        train/<类别名>/*.jpg
        test/<类别名>/*.jpg

增强策略的考虑：
- 随机翻转 / ±15° 旋转：皮肤病灶没有固定方向，手机拍摄角度多变；
- ColorJitter：模拟不同手机、不同光照下的色彩与对比度差异；
- 验证/测试集只做 Resize + CenterCrop + 归一化，保证评估口径一致。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import IMAGENET_MEAN, IMAGENET_STD


def build_transforms() -> Dict[str, transforms.Compose]:
    """训练集（带数据增强）与验证/测试集（仅标准预处理）的变换。"""
    train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((-15, 15)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return {"train": train, "val": val}


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """构建训练 / 测试 DataLoader。

    Returns:
        (dataloaders, sizes)：键均为 "train" / "test"。
    """
    transforms_map = build_transforms()
    image_datasets = {
        phase: datasets.ImageFolder(f"{data_dir}/{phase}", transforms_map[phase])
        for phase in ("train", "test")
    }
    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle=(phase == "train"),
            drop_last=(phase == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for phase in ("train", "test")
    }
    sizes = {phase: len(ds) for phase, ds in image_datasets.items()}
    return dataloaders, sizes
