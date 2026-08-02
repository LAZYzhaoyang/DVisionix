# -*- coding: utf-8 -*-
"""检测任务与网格检测模型的单元测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from dvisionix.data import detection_collate
from dvisionix.models import GridDetectionModel
from dvisionix.training import DetectionTask


def _make_batch(num_classes=3, image_size=64, bs=2):
    samples = []
    for i in range(bs):
        samples.append(
            {
                "image": torch.randn(3, image_size, image_size),
                "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0], [30.0, 30.0, 50.0, 50.0]]),
                "labels": torch.tensor([i % num_classes, (i + 1) % num_classes]),
            }
        )
    return detection_collate(samples)


def test_grid_model_output_shape():
    model = GridDetectionModel(num_classes=3)
    out = model(torch.randn(2, 3, 64, 64))
    # stride 8 -> 64/8 = 8, channels = 5 + 3
    assert out.shape == (2, 8, 8, 8)


def test_detection_collate():
    batch = _make_batch()
    assert batch["image"].shape == (2, 3, 64, 64)
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
    assert isinstance(batch["labels"], list) and len(batch["labels"]) == 2


def test_detection_training_step():
    model = GridDetectionModel(num_classes=3)
    task = DetectionTask(num_classes=3)
    batch = _make_batch()
    device = torch.device("cpu")
    result = task.training_step(model, batch, device)
    assert (
        "loss" in result and "obj_loss" in result and "box_loss" in result and "cls_loss" in result
    )
    assert result["loss"].requires_grad
    # 反向传播可用
    result["loss"].backward()


def test_detection_loss_decreases():
    """在单个固定 batch 上过拟合，损失应下降。"""
    torch.manual_seed(0)
    model = GridDetectionModel(num_classes=3)
    task = DetectionTask(num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _make_batch()
    device = torch.device("cpu")

    first = None
    last = None
    for step in range(20):
        opt.zero_grad()
        out = task.training_step(model, batch, device)
        out["loss"].backward()
        opt.step()
        if step == 0:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


if __name__ == "__main__":
    print("Running detection tests...")
    test_grid_model_output_shape()
    print("ok test_grid_model_output_shape")
    test_detection_collate()
    print("ok test_detection_collate")
    test_detection_training_step()
    print("ok test_detection_training_step")
    test_detection_loss_decreases()
    print("ok test_detection_loss_decreases")
    print("All detection tests passed!")
