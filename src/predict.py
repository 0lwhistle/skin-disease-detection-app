# -*- coding: utf-8 -*-
"""两级推理流水线：人体校验 -> 24 类皮肤病分类，并支持导出 ONNX。

对应 APP 端实际使用的两个模型：

1. ``body_verify``：人体校验模型，过滤非人体照片；
2. ``skin_classifier``：24 类皮肤病分类模型，输出病种与置信度。

``--export-onnx`` 将两个模型导出为 ONNX（输入 1x3x224x224，
输入/输出节点命名为 input/output），交付安卓端做端侧推理。

用法::

    python predict.py --image demo.jpg
    python predict.py --export-onnx --export-dir ../exports
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from classes import CLASSES
from models import build_body_verify_model, build_disease_classifier, get_device


def build_transform():
    return transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def export_onnx(model, export_path):
    """按 APP 端约定的输入输出节点名导出 ONNX。"""
    dummy = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, export_path,
        input_names=["input"], output_names=["output"],
    )
    print(f"ONNX 已导出: {export_path}")


def verify(model, img_tensor):
    """人体校验。

    注意：人体校验头为单输出（Linear(1000, 1)），对单输出做
    softmax/argmax 恒返回类别 0，即本脚本内这一级不会真正拦截
    图片——APP 端按原始输出分数做阈值判断。这里保持与交付模型
    一致的结构，便于复现与对齐。
    """
    with torch.no_grad():
        output = model(img_tensor)
    predicted = torch.argmax(torch.nn.functional.softmax(output, dim=1), 1).item()
    return predicted == 0


def predict(model, img_tensor, top_k=3):
    """皮肤病分类，返回 top-k (病种, 置信度) 列表。"""
    with torch.no_grad():
        probabilities = torch.nn.functional.softmax(model(img_tensor), dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    return [(CLASSES[i], p.item()) for i, p in zip(top_indices.tolist(), top_probs.tolist())]


def parse_args():
    parser = argparse.ArgumentParser(description="皮肤病检测两级推理 / ONNX 导出")
    parser.add_argument("--image", help="待推理图片路径")
    parser.add_argument("--body-weights", default="checkpoints/body_verify_last.pth")
    parser.add_argument("--disease-weights", default="checkpoints/skin_classifier_best.pth")
    parser.add_argument("--export-onnx", action="store_true", help="导出两个模型为 ONNX")
    parser.add_argument("--export-dir", default="exports", help="ONNX 导出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"使用设备: {device}")

    # 人体校验模型
    body_model = build_body_verify_model(pretrained=False).to(device)
    body_model.load_state_dict(torch.load(args.body_weights, map_location=device))
    body_model.eval()

    # 皮肤病分类模型
    disease_model = build_disease_classifier(pretrained=False).to(device)
    disease_model.load_state_dict(torch.load(args.disease_weights, map_location=device))
    disease_model.eval()

    if args.export_onnx:
        os.makedirs(args.export_dir, exist_ok=True)
        export_onnx(body_model.cpu(), os.path.join(args.export_dir, "body_verify.onnx"))
        export_onnx(disease_model.cpu(), os.path.join(args.export_dir, "skin_classifier_24cls.onnx"))

    if not args.image:
        return

    img_tensor = build_transform()(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    if not verify(body_model, img_tensor):
        print("请输入正常清晰的人体皮肤图片！")
        return

    results = predict(disease_model, img_tensor)
    print("预测结果（top-3）:")
    for name, prob in results:
        print(f"  {name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
