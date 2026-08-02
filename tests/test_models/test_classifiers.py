# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分类组合模型 / 分类头 / 检测脚手架测试（S2）。
"""分类组合模型 / 分类头 / 检测脚手架测试（S2）。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import SingleStageDetector, build_model
from dvisionix.models.classifiers import LinearClassifier
from dvisionix.models.heads import ArcFaceHead, ClsHead, MultiLabelHead
from dvisionix.models.losses import BinaryCrossEntropy
from dvisionix.registry import HEADS, MODELS

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]


def test_linear_classifier_default_head():
    model = LinearClassifier(
        backbone={"type": "sequential_backbone", "stages": STAGES, "features_only": False},
        num_classes=10,
    )
    assert isinstance(model.head, ClsHead)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 10)


def test_linear_classifier_build_from_registry():
    m = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES},
            "num_classes": 7,
        }
    )
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 7)


def test_linear_classifier_with_arcface_head():
    model = LinearClassifier(
        backbone={"type": "sequential_backbone", "stages": STAGES},
        head={"type": "arcface", "num_classes": 10},
    )
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)
    assert out.abs().max() <= 30.0 + 1e-5  # s=30


def test_arcface_margin_with_labels():
    head = ArcFaceHead(in_channels=16, num_classes=5)
    x = torch.randn(4, 16)
    labels = torch.randint(0, 5, (4,))
    head.train()
    logits = head(x, labels=labels)
    assert logits.shape == (4, 5)


def test_multi_label_head_and_loss():
    head = MultiLabelHead(in_channels=16, num_classes=5)
    logits = head(torch.randn(4, 16))
    assert logits.shape == (4, 5)
    loss = BinaryCrossEntropy()(logits, torch.randint(0, 2, (4, 5)))
    assert torch.isfinite(loss)


def test_head_registries():
    assert "cls_head" in HEADS and "linear_cls_head" in HEADS
    assert "arcface" in HEADS and "multi_label" in HEADS
    assert "LinearClassifier" in MODELS and "linear_classifier" in MODELS


def test_single_stage_detector_scaffold():
    det = SingleStageDetector(
        backbone={"type": "sequential_backbone", "stages": STAGES, "features_only": True},
        neck={"type": "fpn", "out_channels": 32},
        head={"type": "det_head", "num_classes": 3},
    )
    out = det(torch.randn(1, 3, 64, 64))  # det_head 取最后一层 1/8
    assert out.shape[1] == 5 + 3
    assert det.num_classes == 3
    with pytest.raises(NotImplementedError):
        det.decode(out, (64, 64))
