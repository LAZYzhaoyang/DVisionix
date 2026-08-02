# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: SegmentationModel 与分割头测试（S4）。
"""SegmentationModel 与分割头测试（S4）。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import build_model
from dvisionix.registry import HEADS, MODELS

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}


def _seg_model(head):
    return build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": BACKBONE,
            "head": head,
        }
    )


def test_deeplabv3_head_forward():
    model = _seg_model({"type": "deeplabv3_head", "num_classes": 4})
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 4, 128, 128)


def test_fcn_head_forward():
    model = _seg_model({"type": "fcn_head", "num_classes": 4})
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 4, 128, 128)


def test_seg_head_forward():
    model = _seg_model({"type": "seg_head", "num_classes": 4})
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 4, 128, 128)


def test_unet_decoder_forward():
    model = _seg_model({"type": "unet_decoder", "num_classes": 4})
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 4, 128, 128)


def test_segmentation_registry():
    assert "SegmentationModel" in MODELS and "segmentation_model" in MODELS
    for name in ("seg_head", "fcn_head", "deeplabv3_head", "unet_decoder"):
        assert name in HEADS


def test_segmentation_task_integration():
    from dvisionix.training import SegmentationTask

    task = SegmentationTask(num_classes=4)
    model = _seg_model({"type": "deeplabv3_head", "num_classes": 4})
    batch = {
        "image": torch.randn(2, 3, 128, 128),
        "mask": torch.randint(0, 4, (2, 128, 128)),
    }
    result = task.training_step(model, batch, torch.device("cpu"))
    assert "loss" in result and result["loss"].requires_grad
