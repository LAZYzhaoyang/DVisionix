# -*- coding: utf-8 -*-
"""阶段 A-D 新增模型测试：YOLO / DETR / SegFormer / MaskFormer / 度量学习头 / MultiLabelTask。"""

import pytest
import torch

torch = pytest.importorskip("torch")

from dvisionix.models import build_model
from dvisionix.models.losses import YOLOLoss, DETRLoss
from dvisionix.models.heads import CosFaceHead, SphereFaceHead, AdaFaceHead
from dvisionix.training import MultiLabelTask
from dvisionix.registry import MODELS, HEADS, TASKS
from dvisionix.data import detection_collate

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}
NECK = {"type": "fpn", "out_channels": 32}


def _batch(bs=2, size=128):
    samples = [
        {"image": torch.randn(3, size, size),
         "boxes": torch.tensor([[20., 20., 50., 50.], [70., 70., 100., 100.]]),
         "labels": torch.tensor([0, 1])}
        for _ in range(bs)
    ]
    return detection_collate(samples)


def test_yolo_registered_and_forward_decode():
    model = build_model({"type": "yolo", "num_classes": 3, "backbone": BACKBONE, "neck": NECK,
                         "head": {"type": "yolo_head", "num_classes": 3, "strides": [2, 4, 8]}})
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"cls", "reg"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


def test_yolo_loss_decreases():
    torch.manual_seed(0)
    model = build_model({"type": "yolo", "num_classes": 3, "backbone": BACKBONE, "neck": NECK,
                         "head": {"type": "yolo_head", "num_classes": 3, "strides": [2, 4, 8]}})
    loss_fn = YOLOLoss(num_classes=3, strides=(2, 4, 8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch()
    first = last = None
    for _ in range(10):
        opt.zero_grad()
        out = loss_fn(model(batch["image"]), batch, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_detr_registered_forward_decode():
    model = build_model({"type": "detr", "num_classes": 3, "backbone": BACKBONE,
                         "head": {"type": "detr_head", "num_classes": 3, "d_model": 64, "num_queries": 10,
                                  "num_encoder_layers": 2, "num_decoder_layers": 2, "num_heads": 4}})
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"logits", "boxes"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


def test_detr_loss_decreases():
    torch.manual_seed(1)
    model = build_model({"type": "detr", "num_classes": 3, "backbone": BACKBONE,
                         "head": {"type": "detr_head", "num_classes": 3, "d_model": 64, "num_queries": 10,
                                  "num_encoder_layers": 2, "num_decoder_layers": 2, "num_heads": 4}})
    loss_fn = DETRLoss(num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = _batch()
    first = last = None
    for _ in range(8):
        opt.zero_grad()
        out = loss_fn(model(batch["image"]), batch, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_segformer_and_maskformer_heads():
    seg = build_model({"type": "segmentation_model", "num_classes": 4, "backbone": BACKBONE,
                       "head": {"type": "segformer_head", "num_classes": 4, "channels": 32}})
    assert seg(torch.randn(1, 3, 128, 128)).shape == (1, 4, 128, 128)
    mf = build_model({"type": "segmentation_model", "num_classes": 4, "backbone": BACKBONE,
                      "head": {"type": "maskformer_head", "num_classes": 4, "d_model": 32, "num_queries": 8}})
    assert mf(torch.randn(1, 3, 128, 128)).shape == (1, 4, 128, 128)


def test_metric_learning_heads():
    for head_cls in (CosFaceHead, SphereFaceHead, AdaFaceHead):
        head = head_cls(in_channels=16, num_classes=5)
        logits = head(torch.randn(4, 16), labels=torch.randint(0, 5, (4,)))
        assert logits.shape == (4, 5)


def test_multilabel_task():
    task = MultiLabelTask(num_classes=5)
    from dvisionix.models import LinearClassifier
    model = LinearClassifier(backbone={"type": "sequential_backbone", "stages": STAGES},
                             head={"type": "multi_label", "num_classes": 5})
    batch = {"image": torch.randn(2, 3, 128, 128), "label": torch.randint(0, 2, (2, 5))}
    result = task.training_step(model, batch, torch.device("cpu"))
    assert "loss" in result and result["loss"].requires_grad


def test_v05_registries():
    assert "yolo" in MODELS and "detr" in MODELS
    assert "segformer_head" in HEADS and "maskformer_head" in HEADS
    assert "cosface" in HEADS and "sphereface" in HEADS and "adaface" in HEADS
    assert "MultiLabelTask" in TASKS