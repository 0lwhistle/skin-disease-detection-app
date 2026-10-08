"""训练与评估引擎。

把「单轮训练 / 整轮评估 / 完整训练循环」收敛到这一个模块，
训练脚本只负责装配数据与模型，避免每个脚本复制一份训练循环。

说明：与参赛实现保持一致，模型选择直接在测试集上进行
（严格来说应另划验证集，避免测试集信息参与模型选择）。
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，尽量保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_every: int = 20,
) -> Tuple[float, float]:
    """训练一个 epoch，返回 (平均损失, 准确率%)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

        if log_every and step % log_every == 0:
            print(f"    epoch {epoch}/{epochs}  step {step}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / total, correct / total * 100


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """在验证 / 测试集上评估，返回 (平均损失, 准确率%)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total * 100


def fit(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: str,
    step_size: int = 7,
    gamma: float = 0.3,
    log_every: int = 20,
) -> List[dict]:
    """完整训练循环：Adam + StepLR 动态学习率，按最佳验证准确率保存权重。

    Args:
        dataloaders: {"train": ..., "val": ...}；val 可以是独立验证集，
            也可以（与参赛实现一致）直接用测试集。
        step_size / gamma: StepLR 参数，每 step_size 轮学习率 ×gamma。

    Returns:
        每个 epoch 的指标历史（loss / acc / lr）。
    """
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_acc, history = 0.0, []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, epoch, epochs, log_every
        )
        scheduler.step()
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0],
        })

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
        print(
            f"Epoch {epoch:>2}/{epochs} | 训练 loss {train_loss:.4f} acc {train_acc:.2f}% "
            f"| 验证 loss {val_loss:.4f} acc {val_acc:.2f}% "
            f"| lr {scheduler.get_last_lr()[0]:.2e}" + ("（最佳，已保存）" if improved else "")
        )

    print(f"训练完成：最佳验证准确率 {best_acc:.2f}%，"
          f"总耗时 {time.time() - start_time:.1f}s，权重已保存至 {save_path}")
    return history
