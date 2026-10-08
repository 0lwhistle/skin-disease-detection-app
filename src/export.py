"""ONNX 导出：把 PyTorch 权重转成安卓端约定的 ONNX 模型。

导出约定（与 APP 端一致）：
- 输入形状固定 1x3x224x224；
- 输入 / 输出节点名统一为 ``input`` / ``output``，方便移动端按名取用。

用法::

    python -m src.export --export-dir exports
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .classes import NUM_CLASSES
from .models import build_body_verify_model, build_disease_classifier


def export_onnx(model: nn.Module, export_path: str | Path, opset: int = 11) -> None:
    """将单个模型导出为 ONNX。"""
    model = model.cpu().eval()
    dummy = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(export_path),
        input_names=["input"], output_names=["output"],
        opset_version=opset,
        dynamo=False,  # 固定走稳定的 TorchScript 导出路径：与参赛交付的导出行为一致，也不依赖 onnxscript
    )
    print(f"ONNX 已导出: {export_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="导出 APP 端两个模型为 ONNX")
    parser.add_argument("--body-weights", default="checkpoints/body_verify_best.pth")
    parser.add_argument("--disease-weights", default="checkpoints/skin_classifier_best.pth")
    parser.add_argument("--export-dir", type=Path, default=Path("exports"))
    args = parser.parse_args()

    args.export_dir.mkdir(parents=True, exist_ok=True)

    body_model = build_body_verify_model(pretrained=False)
    body_model.load_state_dict(torch.load(args.body_weights, map_location="cpu"))
    export_onnx(body_model, args.export_dir / "body_verify.onnx")

    disease_model = build_disease_classifier(pretrained=False)
    disease_model.load_state_dict(torch.load(args.disease_weights, map_location="cpu"))
    export_onnx(disease_model, args.export_dir / f"skin_classifier_{NUM_CLASSES}cls.onnx")


if __name__ == "__main__":
    main()
