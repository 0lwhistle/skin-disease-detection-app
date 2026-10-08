"""两级推理流水线：人体校验 → 24 类病种分类。

对应 APP 的端侧推理流程：
1. 人体校验模型先过滤非人体 / 非皮肤照片（二分类 softmax 阈值判断）；
2. 通过后再由 24 类分类模型输出病种与置信度（top-k）。

用法::

    python -m src.predict --image demo.jpg
    python -m src.predict --image-dir imgs/ --body-threshold 0.5

ONNX 导出见 ``python -m src.export``。
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .classes import CLASSES
from .datasets import build_transforms
from .models import build_body_verify_model, build_disease_classifier, get_device

# 类别索引与训练数据目录的字母序一致（见 train_body_verify 模块说明）
BODY_CLASSES = ("human body", "not human body")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify(model: torch.nn.Module, tensor: torch.Tensor, threshold: float = 0.5) -> tuple:
    """人体校验，返回 (是否人体, 人体概率)。

    阈值按业务可调：宁缺勿滥（减少误分类）就调高，
    优先保证能识别就调低。
    """
    probs = torch.softmax(model(tensor), dim=1)[0]
    body_prob = probs[BODY_CLASSES.index("human body")].item()
    return body_prob >= threshold, body_prob


def predict(model: torch.nn.Module, tensor: torch.Tensor, top_k: int = 3) -> list:
    """病种分类，返回 top-k 的 [(类别名, 置信度)] 列表。"""
    probs = torch.softmax(model(tensor), dim=1)[0]
    top_probs, top_indices = torch.topk(probs, k=top_k)
    return [(CLASSES[i], p.item()) for i, p in zip(top_indices.tolist(), top_probs.tolist())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="皮肤病检测两级推理")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="单张待推理图片路径")
    source.add_argument("--image-dir", help="批量推理图片目录")
    parser.add_argument("--body-weights", default="checkpoints/body_verify_best.pth")
    parser.add_argument("--disease-weights", default="checkpoints/skin_classifier_best.pth")
    parser.add_argument("--body-threshold", type=float, default=0.5, help="人体校验阈值")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"设备: {device}")

    body_model = build_body_verify_model(pretrained=False).to(device)
    body_model.load_state_dict(torch.load(args.body_weights, map_location=device))
    body_model.eval()

    disease_model = build_disease_classifier(pretrained=False).to(device)
    disease_model.load_state_dict(torch.load(args.disease_weights, map_location=device))
    disease_model.eval()

    transform = build_transforms()["val"]
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not image_paths:
            raise SystemExit(f"目录中没有图片: {image_dir}")

    for path in image_paths:
        try:
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        except Exception as e:
            print(f"{path.name}: 读取失败（{e}）")
            continue

        is_body, body_prob = verify(body_model, tensor, args.body_threshold)
        if not is_body:
            print(f"{path.name}: 非人体照片（人体概率 {body_prob:.2%}），已拦截")
            continue

        print(f"{path.name}: 人体校验通过（{body_prob:.2%}），病种预测 top-{args.top_k}:")
        for name, prob in predict(disease_model, tensor, args.top_k):
            print(f"    {name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
