# -*- coding: utf-8 -*-
"""人体校验模型批量测试。

遍历指定目录下的图片，逐张输出人体校验结果，用于人工核对
该级过滤器的召回情况。

用法::

    python test_body_verify.py --weights checkpoints/body_verify_last.pth --image-dir ../data/test-imgs
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from models import build_body_verify_model, get_device

CLASSES_NAME = ["human body", "not human body"]


def build_transform():
    return transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="人体校验模型批量测试")
    parser.add_argument("--weights", default="checkpoints/body_verify_last.pth")
    parser.add_argument("--image-dir", required=True, help="待测试图片目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    model = build_body_verify_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    transform = build_transform()
    for img_name in os.listdir(args.image_dir):
        img_path = os.path.join(args.image_dir, img_name)
        try:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except Exception as e:
            print(f"{img_name}: 读取失败（{e}）")
            continue
        with torch.no_grad():
            output = model(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, 1).item()
        print(f"{img_name}: {CLASSES_NAME[predicted]}")


if __name__ == "__main__":
    main()
