"""YOLO11-seg 人体实例分割 + CNN 皮肤病分类增强流水线。

针对多人合影 / 复杂背景场景的增强推理方案：

1. YOLO11-seg 分割出画面中的每个人体实例；
2. 掩码外区域置白（避免背景干扰 CNN），按检测框裁出皮肤区域；
3. 皮肤区域逐人送入 24 类皮肤病分类模型；
4. 在原图上叠加半透明掩码、检测框与预测标签，输出结果图与文本报告。

用法（仓库根目录）::

    python -m yolo_pipeline.skin_analysis --image test1.jpg \
        --cnn-weights checkpoints/skin_classifier_best.pth

加载参赛时期的旧结构权重（如 model_test/cnn_model）时加 ``--legacy-head``。
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能保存结果图
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classes import CLASSES
from src.models import build_disease_classifier

# 可视化用色调盘（多人时循环取色）
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
]


class SkinAnalysisSystem:
    """YOLO 人体分割 + CNN 病种分类的两级分析系统。"""

    def __init__(self, cnn_weights: str, yolo_weights: str = "yolo11n-seg.pt",
                 conf: float = 0.5, legacy_head: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"设备: {self.device}")

        self.cnn_model = build_disease_classifier(pretrained=False, legacy_head=legacy_head)
        self.cnn_model.load_state_dict(torch.load(cnn_weights, map_location=self.device))
        self.cnn_model.to(self.device).eval()
        print(f"CNN 分类模型加载完成: {cnn_weights}")

        self.yolo_model = YOLO(yolo_weights)
        print(f"YOLO 分割模型加载完成: {yolo_weights}")

        # 与分类模型验证预处理保持一致（284 延续参赛时的测试口径）
        self.cnn_transform = transforms.Compose([
            transforms.Resize([284, 284]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.conf = conf

    # ------------------------------------------------------------------
    def extract_skin_regions(self, image_path: str):
        """分割人体并提取各人的皮肤区域。

        Returns:
            (skin_regions, bboxes, masks, orig_image)，未读到图时前
            三项为空列表、orig_image 为 None；未检测到人体时列表为空。
        """
        orig_image = cv2.imread(image_path)
        if orig_image is None:
            print(f"无法读取图像: {image_path}")
            return [], [], [], None

        orig_h, orig_w = orig_image.shape[:2]
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(image_rgb, conf=self.conf)

        skin_regions, bboxes, masks = [], [], []
        if results[0].masks is None:
            return skin_regions, bboxes, masks, orig_image

        for i, (mask, box) in enumerate(zip(results[0].masks.data, results[0].boxes.xyxy)):
            if results[0].boxes.cls[i].item() != 0:  # 只保留人体类别（COCO class 0 = person）
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            mask_np = mask.cpu().numpy()
            # YOLO 掩码输出分辨率与原图不一致时先对齐，再按阈值二值化
            if mask_np.shape != (orig_h, orig_w):
                mask_np = cv2.resize(mask_np, (orig_w, orig_h))
            mask_binary = mask_np > 0.5

            # 掩码外区域置白：CNN 只见过皮肤病灶主体，背景杂物会干扰判断
            skin_region = np.ones_like(orig_image) * 255
            skin_region[mask_binary] = orig_image[mask_binary]
            cropped = skin_region[y1:y2, x1:x2]

            if cropped.size > 0:
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                skin_regions.append(Image.fromarray(cropped_rgb))
                bboxes.append((x1, y1, x2, y2))
                masks.append(mask_binary)

        return skin_regions, bboxes, masks, orig_image

    # ------------------------------------------------------------------
    def cnn_predict(self, pil_image: Image.Image):
        """对单张皮肤区域分类，返回 (类别号, 置信度, 全类别概率数组)。"""
        tensor = self.cnn_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.cnn_model(tensor), dim=1)[0]
        pred = int(torch.argmax(probs).item())
        return pred, probs[pred].item(), probs.cpu().numpy()

    # ------------------------------------------------------------------
    def visualize_on_original(self, orig_image: np.ndarray, bboxes, masks, predictions) -> np.ndarray:
        """在原图上叠加半透明掩码、检测框与预测标签。"""
        result = orig_image.copy()
        orig_h, orig_w = result.shape[:2]

        for i, ((x1, y1, x2, y2), mask, pred) in enumerate(zip(bboxes, masks, predictions)):
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h)) > 0.5

            color = COLORS[i % len(COLORS)]
            color_bgr = (color[2], color[1], color[0])

            colored = np.zeros_like(result, dtype=np.uint8)
            colored[np.stack([mask] * 3, axis=-1)] = color_bgr
            cv2.addWeighted(colored, 0.3, result, 0.7, 0, result)

            cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 3)
            label = f"P{i + 1}: {pred['class_name']} ({pred['confidence']:.1%})"
            cv2.putText(result, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        return result

    # ------------------------------------------------------------------
    def process(self, image_path: str, save_dir: str = "results"):
        """完整流程：分割 → 提取 → 分类 → 可视化 → 保存结果。"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("步骤1: YOLO 人体分割与皮肤区域提取")
        skin_regions, bboxes, masks, orig_image = self.extract_skin_regions(image_path)
        if orig_image is None:
            return []
        if not skin_regions:
            print("未检测到人体")
            cv2.imwrite(str(Path(save_dir) / "no_human_detected.jpg"), orig_image)
            return []
        print(f"检测到 {len(skin_regions)} 个人体")

        print("步骤2: CNN 皮肤病分类")
        predictions = []
        for i, (region, bbox, mask) in enumerate(zip(skin_regions, bboxes, masks)):
            region.save(str(Path(save_dir) / f"person_{i + 1}_skin_region.jpg"))
            pred_class, confidence, all_probs = self.cnn_predict(region)
            top3 = np.argsort(all_probs)[-3:][::-1]
            predictions.append({
                "person_id": i + 1,
                "bbox": bbox,
                "class_name": CLASSES[pred_class],
                "confidence": confidence,
                "top3": [(CLASSES[int(idx)], float(all_probs[idx])) for idx in top3],
            })
            print(f"  人体 {i + 1}: {predictions[-1]['class_name']} ({confidence:.2%})")

        print("步骤3: 生成可视化与结果报告")
        result_image = self.visualize_on_original(orig_image, bboxes, masks, predictions)
        cv2.imwrite(str(Path(save_dir) / "result_on_original.jpg"), result_image)

        fig = plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("YOLO Segmentation + CNN Prediction", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(str(Path(save_dir) / "result_detail.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        report_path = Path(save_dir) / "results.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"人体皮肤分析结果（{len(predictions)} 人）\n")
            f.write("=" * 40 + "\n")
            for pred in predictions:
                f.write(f"\n人体 {pred['person_id']}  边界框 {pred['bbox']}\n")
                f.write(f"  预测: {pred['class_name']}  置信度: {pred['confidence']:.2%}\n")
                for name, prob in pred["top3"]:
                    f.write(f"    - {name}: {prob:.2%}\n")

        print(f"结果已保存至: {save_dir}")
        return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 分割 + CNN 病种分类增强流水线")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--cnn-weights", default="checkpoints/skin_classifier_best.pth",
                        help="皮肤病 CNN 权重（本仓库训练得到）")
    parser.add_argument("--yolo-weights", default="yolo11n-seg.pt",
                        help="YOLO11-seg 权重（ultralytics 官方下载）")
    parser.add_argument("--legacy-head", action="store_true",
                        help="CNN 权重为参赛时期旧结构（如 model_test/cnn_model）时开启")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 检测置信度阈值")
    parser.add_argument("--save-dir", default="results", help="结果保存目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = SkinAnalysisSystem(
        args.cnn_weights, args.yolo_weights, conf=args.conf, legacy_head=args.legacy_head
    )
    system.process(args.image, args.save_dir)


if __name__ == "__main__":
    main()
