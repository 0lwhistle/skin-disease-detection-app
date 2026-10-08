"""皮肤病 24 分类模型训练入口。

同一入口覆盖两阶段训练，用 ``--init-from`` 区分：

第一阶段（全量训练，ImageNet 预训练 + 迁移学习）::

    python -m src.train --data-dir data/skin-disease

第二阶段（加载第一阶段最佳权重，低学习率微调收敛）::

    python -m src.train --data-dir data/skin-disease \
        --init-from checkpoints/skin_classifier_best.pth \
        --epochs 5 --lr 1e-6 --step-size 2 \
        --save-name skin_classifier_finetuned_best.pth
"""

import argparse
from pathlib import Path

import torch

from .classes import CLASSES
from .datasets import build_dataloaders
from .engine import fit, seed_everything
from .models import build_disease_classifier, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="皮肤病 24 分类模型训练（含微调模式）")
    parser.add_argument("--data-dir", required=True, help="ImageFolder 数据根目录（含 train/ test/）")
    parser.add_argument("--init-from", default=None, help="已有权重路径；传入即进入低学习率微调模式")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率（微调建议 1e-6）")
    parser.add_argument("--step-size", type=int, default=7, help="StepLR 步长（微调建议 2）")
    parser.add_argument("--gamma", type=float, default=0.3, help="StepLR 衰减系数")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-name", default="skin_classifier_best.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"设备: {device}")

    dataloaders, sizes = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    print(f"训练集 {sizes['train']} 张 / 测试集 {sizes['test']} 张，共 {len(CLASSES)} 类")

    # 类别顺序校验：ImageFolder 按目录名字母序编号，与 classes.py 不一致
    # 会导致训练标签与推理类别错位，这里提前拦截
    dataset_classes = dataloaders["train"].dataset.classes
    if dataset_classes != list(CLASSES):
        print("警告：数据集类别顺序与 classes.py 不一致，推理时类别会错位！")
        print(f"  数据集类别: {dataset_classes}")

    finetune = args.init_from is not None
    model = build_disease_classifier(pretrained=not finetune)
    if finetune:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"已加载权重，进入低学习率微调模式: {args.init_from}")
    model.to(device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    fit(
        model,
        dataloaders={"train": dataloaders["train"], "val": dataloaders["test"]},
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=str(args.save_dir / args.save_name),
        step_size=args.step_size,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
