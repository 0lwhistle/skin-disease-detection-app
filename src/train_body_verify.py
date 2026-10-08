"""人体校验模型（人体 / 非人体二分类）训练。

该模型是 APP 两级推理的第一级：先过滤非人体照片，再交给病种
分类模型，降低误分类率。参赛原始实现为单输出打分头且无验证集
（逐轮覆盖保存）；重构版规范化为标准二分类头，并随机划出验证集、
按最佳验证准确率保存权重。

类别编号按数据目录名的字母序生成，推理侧 ``predict.BODY_CLASSES``
需与其保持一致::

    data_dir/
        human_body/      -> 类别 0
        not_human_body/  -> 类别 1

用法::

    python -m src.train_body_verify --data-dir data/body-verify --epochs 3
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from .datasets import build_transforms
from .engine import fit, seed_everything
from .models import build_body_verify_model, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人体校验二分类模型训练")
    parser.add_argument("--data-dir", required=True, help="ImageFolder 数据目录（human_body / not_human_body）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="从训练数据中划出做验证的比例")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-name", default="body_verify_best.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"设备: {device}")

    transforms_map = build_transforms()
    # 训练 / 验证需要不同变换（验证不做增强），而 random_split 的两个子集
    # 共享同一个底层 dataset，因此用相同种子对两个 ImageFolder 各切一次，
    # 保证两边拿到的是同一批图片、各自的变换
    train_full = datasets.ImageFolder(args.data_dir, transforms_map["train"])
    val_full = datasets.ImageFolder(args.data_dir, transforms_map["val"])
    n_val = int(len(train_full) * args.val_fraction)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = random_split(train_full, [n_train, n_val], generator=generator)
    train_set = Subset(train_full, train_indices.indices)
    val_set = Subset(val_full, val_indices.indices)
    print(f"类别映射: {dict(zip(train_full.class_to_idx.values(), train_full.class_to_idx.keys()))}")
    print(f"训练集 {len(train_set)} 张 / 验证集 {len(val_set)} 张")

    dataloaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
    }

    model = build_body_verify_model().to(device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    fit(
        model,
        dataloaders=dataloaders,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=str(args.save_dir / args.save_name),
    )


if __name__ == "__main__":
    main()
