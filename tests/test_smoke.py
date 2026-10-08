"""冒烟测试：不依赖真实权重与数据集，验证模型结构与导出链路。

运行方式（仓库根目录）::

    python -m pytest tests -q      # pytest
    python -m tests.test_smoke     # 无 pytest 时的直跑模式
"""

import tempfile
from pathlib import Path

import torch

from src.classes import NUM_CLASSES
from src.export import export_onnx
from src.models import build_body_verify_model, build_disease_classifier


def test_disease_classifier_output_shape():
    """24 类分类模型：前向输出形状应为 (1, 24)。"""
    model = build_disease_classifier(pretrained=False).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 224, 224))
    assert out.shape == (1, NUM_CLASSES)


def test_body_verify_output_shape():
    """人体校验模型：前向输出形状应为 (1, 2)。"""
    model = build_body_verify_model(pretrained=False).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_legacy_head_compatible():
    """旧结构开关：叠加头版本应能对齐参赛时期的网络形状。"""
    model = build_disease_classifier(pretrained=False, legacy_head=True)
    names = [name for name, _ in model.named_parameters()]
    assert "fc.weight" in names and "add_linear.weight" in names


def test_onnx_export_produces_file():
    """ONNX 导出链路：能生成非空模型文件。"""
    model = build_disease_classifier(pretrained=False).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "skin.onnx"
        export_onnx(model, path)
        assert path.exists() and path.stat().st_size > 0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"{name} 通过")
    print(f"共 {len(tests)} 项冒烟测试全部通过")
