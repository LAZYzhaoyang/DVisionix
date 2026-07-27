# -*- coding: utf-8 -*-
"""
DVisionix ONNX 导出 Demo

演示如何将模型导出为 ONNX 格式并验证精度。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dvisionix.models import SimpleCNN
from dvisionix.export import ONNXExporter


def main() -> None:
    print("=" * 60)
    print("DVisionix ONNX Export Demo")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)

    ckpt_path = "./checkpoints/cifar10/best_model.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        print(f"[OK] Loaded weights: {ckpt_path}")
    else:
        print("[WARN] No trained weights found, use random init (demo only)")

    exporter = ONNXExporter(model=model, input_shape=(3, 32, 32), device="cpu")
    output_path = "./exports/simple_cnn.onnx"
    exporter.export(output_path=output_path, dynamic_batch=True, opset_version=17)
    exporter.verify(output_path, num_samples=3)

    print("=" * 60)
    print(f"[OK] Done. Exported: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
