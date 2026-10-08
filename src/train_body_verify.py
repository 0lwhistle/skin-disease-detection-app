# -*- coding: utf-8 -*-
"""人体校验模型训练（皮肤病检测 APP 的第一级过滤）。

在 APP 的推理流程中，先用该模型判断照片是否为清晰的人体皮肤照，
过滤非人体图片后再交给 24 类病种分类模型，降低误分类率。

说明：与交付权重、安卓端 ONNX 保持一致，本模型采用单输出打分头
（``Linear(1000, 1)``），原始实现按分类方式训练，这里保持原样。

数据目录采用 ImageFolder 格式（两个子目录）::

    data_dir/
        human_body/      # 人体皮肤照
        not_human_body/  # 非人体图片

用法::

    python train_body_verify.py --data-dir ../data/body-verify --epochs 3
"""

import argparse
import os
import time

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_body_verify_model, get_device


def build_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="人体校验模型训练")
    parser.add_argument("--data-dir", required=True, help="ImageFolder 数据目录（人体 / 非人体两个子目录）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-name", default="body_verify_last.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"使用设备: {device}")

    dataset = datasets.ImageFolder(args.data_dir, build_transform())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"数据量: {len(dataset)} 张，类别: {dataset.classes}")

    model = build_body_verify_model().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, correct = 0.0, 0
        for step, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (output.argmax(1) == label).sum().item()
            if (step + 1) % 20 == 0:
                print(f"  epoch {epoch + 1}  batch {step + 1}/{len(dataloader)}  loss={loss.item():.4f}")
        print(
            f"第 {epoch + 1}/{args.epochs} 轮结束  "
            f"loss={epoch_loss / len(dataset):.5f}  acc={correct / len(dataset) * 100:.2f}%"
        )
        torch.save(model.state_dict(), save_path)

    print(f"训练完成，总耗时 {time.time() - start_time:.1f}s，权重已保存: {save_path}")


if __name__ == "__main__":
    main()
