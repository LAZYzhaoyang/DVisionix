# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: FCOS / RetinaNet 检测器、assigner、检测损失测试（S3）。
"""FCOS / RetinaNet 检测器、assigner、检测损失测试（S3）。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.models import FCOSDetector, RetinaNetDetector, build_model
from dvisionix.models.detectors import AnchorGenerator
from dvisionix.models.losses import (
    ATSSAssigner,
    FCOSAssigner,
    FCOSDetectionLoss,
    MaxIoUAssigner,
    RetinaNetLoss,
)

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}
NECK = {"type": "fpn", "out_channels": 32}
STRIDES = (2, 4, 8)
SCALES = (0.0, 24.0, 48.0, 96.0, 1e10)


def _fcos_model():
    return build_model(
        {
            "type": "fcos",
            "num_classes": 3,
            "backbone": BACKBONE,
            "neck": NECK,
            "head": {"type": "fcos_head", "num_classes": 3, "strides": list(STRIDES)},
        }
    )


def _retinanet_model():
    return build_model(
        {
            "type": "retinanet",
            "num_classes": 3,
            "backbone": BACKBONE,
            "neck": NECK,
            "head": {"type": "retinanet_head", "num_classes": 3, "num_anchors": 9},
            "strides": list(STRIDES),
            "base_sizes": [8, 16, 32],
        }
    )


def _batch(num_classes=3, bs=2, size=128):
    samples = []
    for i in range(bs):
        samples.append(
            {
                "image": torch.randn(3, size, size),
                "boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0], [70.0, 70.0, 100.0, 100.0]]),
                "labels": torch.tensor([i % num_classes, (i + 1) % num_classes]),
            }
        )
    return detection_collate(samples)


def test_anchor_generator_shapes():
    gen = AnchorGenerator(strides=STRIDES, base_sizes=(8, 16, 32))
    feats = [torch.randn(1, 4, 64, 64), torch.randn(1, 4, 32, 32), torch.randn(1, 4, 16, 16)]
    anchors = gen.grid_anchors(feats)
    assert gen.num_anchors == 9
    assert anchors[0].shape == (64 * 64 * 9, 4)
    assert anchors[2].shape == (16 * 16 * 9, 4)


def test_fcos_detector_forward_and_decode():
    model = _fcos_model()
    assert isinstance(model, FCOSDetector)
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"cls", "reg", "center"}
    assert len(preds["cls"]) == 3
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4
    assert boxes[0].shape[0] == scores[0].shape[0] == labels[0].shape[0]


def test_retinanet_detector_forward_and_decode():
    model = _retinanet_model()
    assert isinstance(model, RetinaNetDetector)
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"cls", "reg"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


def test_fcos_assigner_basic():
    assigner = FCOSAssigner(num_classes=3, strides=STRIDES, scales=SCALES)
    gt = torch.tensor([[20.0, 20.0, 50.0, 50.0]])
    labels_gt = torch.tensor([1])
    assigned = assigner.assign([(64, 64), (32, 32), (16, 16)], gt, labels_gt, (128, 128))
    lbl0 = assigned[0][0]
    assert (lbl0 > 0).sum() > 0
    assert set(lbl0.unique().tolist()) <= {0, 2}


def test_maxiou_assigner_basic():
    assigner = MaxIoUAssigner(num_classes=3)
    anchors = torch.tensor([[16.0, 16.0, 48.0, 48.0], [100.0, 100.0, 120.0, 120.0]])
    labels, targets = assigner.assign(
        anchors, torch.tensor([[20.0, 20.0, 50.0, 50.0]]), torch.tensor([2]), (128, 128)
    )
    assert labels[0] == 3
    assert labels[1] == 0


def test_atss_assigner_runs():
    assigner = ATSSAssigner(num_classes=3, num_anchors=9, topk=9)
    gen = AnchorGenerator(strides=STRIDES, base_sizes=(8, 16, 32))
    feats = [torch.randn(1, 4, 64, 64), torch.randn(1, 4, 32, 32), torch.randn(1, 4, 16, 16)]
    anchors = gen.grid_anchors(feats)
    gt = torch.tensor([[20.0, 20.0, 50.0, 50.0], [70.0, 70.0, 100.0, 100.0]])
    labels, targets = assigner.assign(anchors, gt, torch.tensor([0, 1]), STRIDES, (128, 128))
    assert labels.shape[0] == sum(a.shape[0] for a in anchors)
    assert (labels > 0).sum() > 0


def test_fcos_loss_decreases():
    torch.manual_seed(0)
    model = _fcos_model()
    loss_fn = FCOSDetectionLoss(num_classes=3, strides=STRIDES, scales=SCALES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch()
    first = last = None
    for step in range(15):
        opt.zero_grad()
        preds = model(batch["image"])
        out = loss_fn(preds, batch, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if step == 0:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_retinanet_loss_decreases():
    torch.manual_seed(1)
    model = _retinanet_model()
    loss_fn = RetinaNetLoss(
        num_classes=3, strides=STRIDES, base_sizes=(8, 16, 32), assigner="max_iou"
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch()
    first = last = None
    for step in range(15):
        opt.zero_grad()
        preds = model(batch["image"])
        out = loss_fn(preds, batch, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if step == 0:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_retinanet_loss_atss_runs():
    model = _retinanet_model()
    loss_fn = RetinaNetLoss(num_classes=3, strides=STRIDES, base_sizes=(8, 16, 32), assigner="atss")
    batch = _batch()
    preds = model(batch["image"])
    out = loss_fn(preds, batch, image_hw=(128, 128))
    assert torch.isfinite(out["loss"])
