# -*- coding: utf-8 -*-
"""v0.10.0 方向 3 测试：YOLOv7/v10 / CenterNet / BiSeNet / CircleLoss / SimCLR。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.models import build_model
from dvisionix.models.layers import build_layer
from dvisionix.models.losses import CenterNetLoss, CircleLoss, InfoNCELoss, OneToOneYOLOLoss
from dvisionix.registry import HEADS, LAYERS, LOSSES, MODELS

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}

BATCH = detection_collate(
    [
        {
            "image": torch.randn(3, 128, 128),
            "boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0], [70.0, 70.0, 100.0, 100.0]]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "image": torch.randn(3, 128, 128),
            "boxes": torch.tensor([[30.0, 30.0, 60.0, 60.0]]),
            "labels": torch.tensor([2]),
        },
    ]
)


# ---------- YOLO 系列 ----------


def test_eelan_layer():
    assert "eelan_layer" in LAYERS
    out = build_layer(
        {"type": "eelan_layer", "in_channels": 32, "out_channels": 64, "num_blocks": 2}
    )(torch.randn(1, 32, 16, 16))
    assert out.shape == (1, 64, 16, 16)


def test_yolov10_forward_decode_and_loss():
    assert "yolo_v10" in MODELS and "yolo_v10_head" in HEADS and "yolo_v10_detection" in LOSSES
    model = build_model(
        {
            "type": "yolo_v10",
            "num_classes": 3,
            "backbone": BACKBONE,
            "neck": {"type": "fpn", "out_channels": 64},
            "head": {"type": "yolo_v10_head", "num_classes": 3, "strides": [2, 4, 8]},
        }
    )
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"cls", "reg"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4

    torch.manual_seed(0)
    loss_fn = OneToOneYOLOLoss(num_classes=3, strides=(2, 4, 8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    first = last = None
    for _ in range(8):
        opt.zero_grad()
        out = loss_fn(model(BATCH["image"]), BATCH, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


# ---------- CenterNet ----------


def test_centernet_forward_decode_and_loss():
    assert "centernet" in MODELS and "centernet_head" in HEADS and "centernet_detection" in LOSSES
    bb = {
        "type": "sequential_backbone",
        "stages": [
            {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 4},
            {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
        ],
        "features_only": True,
    }
    model = build_model(
        {
            "type": "centernet",
            "num_classes": 3,
            "backbone": bb,
            "stride": 8,
            "head": {"type": "centernet_head", "num_classes": 3},
        }
    )
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"heatmap", "wh", "offset"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4

    torch.manual_seed(0)
    loss_fn = CenterNetLoss(num_classes=3, stride=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for _ in range(8):
        opt.zero_grad()
        out = loss_fn(model(BATCH["image"]), BATCH, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_new_detection_configs_load():
    from dvisionix.config import Config

    for name in ("yolov7_synthetic", "yolov10_synthetic", "centernet_synthetic"):
        cfg = Config.from_yaml(rf"D:\ZhaoyangProject\DVisionix\configs\detection\{name}.yaml")
        m = build_model(cfg.model.to_dict())
        assert m is not None


# ---------- 分割：BiSeNet ----------


def test_bisenet_head_forward():
    assert "bisenet_head" in HEADS
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": BACKBONE,
            "head": {"type": "bisenet_head", "num_classes": 4},
        }
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


# ---------- 分类：CircleLoss / SimCLR ----------


def test_circle_loss_head_and_loss():
    assert "circle_loss_head" in HEADS and "circle_loss" in LOSSES
    torch.manual_seed(0)
    model = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "circle_loss_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    loss_fn = CircleLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.randint(0, 5, (8,))
    first = last = None
    for _ in range(10):
        opt.zero_grad()
        loss = loss_fn(model(torch.randn(8, 3, 64, 64)), labels)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_simclr_head_and_info_nce():
    assert "simclr_head" in HEADS and "info_nce" in LOSSES
    torch.manual_seed(0)
    model = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "simclr_head", "num_classes": 5, "out_dim": 16},
            "num_classes": 5,
        }
    )
    loss_fn = InfoNCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for _ in range(10):
        opt.zero_grad()
        z1 = model(torch.randn(8, 3, 64, 64))
        z2 = model(torch.randn(8, 3, 64, 64))
        loss = loss_fn(z1, z2)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first
