# -*- coding: utf-8 -*-
"""模型组件化架构测试：backbone / neck / head / GeneralizedModel。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import build_model, build_neck, build_head, NECKS, HEADS, MODELS


def test_fpn_build_and_forward():
    fpn = build_neck({"type": "fpn", "in_channels": [64, 128, 256], "out_channels": 64})
    feats = [torch.randn(1, c, 32 // (2 ** i), 32 // (2 ** i)) for i, c in enumerate([64, 128, 256])]
    out = fpn(feats)
    assert len(out) == 3
    assert all(o.shape[1] == 64 for o in out)


def test_heads_shapes():
    cls_head = build_head({"type": "cls_head", "in_channels": 16, "num_classes": 5})
    assert cls_head(torch.randn(2, 16)).shape == (2, 5)

    seg_head = build_head({"type": "seg_head", "in_channels": 16, "num_classes": 4, "output_size": [8, 8]})
    assert seg_head(torch.randn(2, 16, 4, 4)).shape == (2, 4, 8, 8)

    det_head = build_head({"type": "det_head", "in_channels": 16, "num_classes": 3})
    out = det_head(torch.randn(2, 16, 8, 8))
    assert out.shape[1] == 5 + 3


def test_registries_contain_components():
    assert "fpn" in NECKS
    assert "cls_head" in HEADS
    assert "seg_head" in HEADS
    assert "det_head" in HEADS
    assert "generalized" in MODELS


def test_generalized_classification():
    pytest.importorskip("timm")
    m = build_model({
        "type": "generalized", "task_type": "classification",
        "backbone": {"type": "timm_backbone", "name": "resnet18", "pretrained": False},
        "head": {"type": "cls_head", "num_classes": 7},
    })
    out = m(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 7)


def test_generalized_segmentation():
    pytest.importorskip("timm")
    m = build_model({
        "type": "generalized", "task_type": "segmentation",
        "backbone": {"type": "timm_backbone", "name": "resnet18", "features_only": True,
                     "out_indices": [4], "pretrained": False},
        "head": {"type": "seg_head", "num_classes": 4},
    })
    out = m(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


def test_base_model_contract():
    from dvisionix.models.base import BaseModel, TASK_TYPES
    assert "classification" in TASK_TYPES
    with pytest.raises(ValueError):
        BaseModel(task_type="not_a_task")
