# -*- coding: utf-8 -*-
"""皮肤病 24 分类模型训练（ResNet50 迁移学习，第一阶段全量训练）。

数据目录采用 ImageFolder 格式::

    data_dir/
        train/
            光化性角化病基底细胞癌及恶性病变/
            ...
        test/
            ...（与 train 同 24 类）

训练策略：Adam + StepLR 动态学习率，每个 epoch 在测试集上评估，
按最佳测试准确率保存权重（state_dict）。

用法::

    python train_skin_classifier.py --data-dir ../data/skin-disease --epochs 30
"""

import argparse
import os
import time

from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classes import NUM_CLASSES
from models import build_disease_classifier, get_device


def build_transforms():
    """训练集做数据增强，测试集只做统一预处理。"""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return {
        "train": transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((-15, 15)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            normalize,
        ]),
        "test": transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="皮肤病 24 分类模型训练")
    parser.add_argument("--data-dir", required=True, help="ImageFolder 数据根目录（含 train/ test/）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam 初始学习率")
    parser.add_argument("--save-dir", default="checkpoints", help="权重保存目录")
    parser.add_argument("--save-name", default="skin_classifier_best.pth", help="最佳权重文件名")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"使用设备: {device}")

    transforms_map = build_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), transforms_map[x])
        for x in ("train", "test")
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, drop_last=True)
        for x in ("train", "test")
    }
    data_sizes = {x: len(image_datasets[x]) for x in ("train", "test")}
    print(f"训练集 {data_sizes['train']} 张 / 测试集 {data_sizes['test']} 张，共 {NUM_CLASSES} 类")

    model = build_disease_classifier().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, args.save_name)
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # ---------- 训练 ----------
        model.train()
        train_loss, train_correct = 0.0, 0
        for step, (img, label) in enumerate(dataloaders["train"]):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (output.argmax(1) == label).sum().item()
            if (step + 1) % 20 == 0:
                print(f"  epoch {epoch + 1}  batch {step + 1}/{len(dataloaders['train'])}  loss={loss.item():.4f}")
        scheduler.step()
        print(
            f"第 {epoch + 1}/{args.epochs} 轮训练结束  "
            f"train_loss={train_loss / data_sizes['train']:.4f}  "
            f"train_acc={train_correct / data_sizes['train'] * 100:.2f}%  "
            f"lr={optimizer.state_dict()['param_groups'][0]['lr']:.2e}"
        )

        # ---------- 测试 ----------
        model.eval()
        test_loss, test_correct = 0.0, 0
        with torch.no_grad():
            for img, label in dataloaders["test"]:
                img, label = img.to(device), label.to(device)
                output = model(img)
                test_loss += criterion(output, label).item()
                test_correct += (output.argmax(1) == label).sum().item()
        test_acc = test_correct / data_sizes["test"] * 100
        print(f"第 {epoch + 1} 轮测试结束  test_loss={test_loss / data_sizes['test']:.4f}  test_acc={test_acc:.2f}%")

        # 按最佳测试准确率保存
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"  最佳模型已保存: {best_path}")

    print(f"训练完成，总耗时 {time.time() - start_time:.1f}s，最佳测试准确率 {best_acc:.2f}%")


if __name__ == "__main__":
    main()
