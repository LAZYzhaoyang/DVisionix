# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 模型组件化架构测试：backbone / neck / head 组件与构建入口。
"""模型组件化架构测试：backbone / neck / head 组件与构建入口。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import HEADS, MODELS, NECKS, build_head, build_model, build_neck


def test_fpn_build_and_forward():
    fpn = build_neck({"type": "fpn", "in_channels": [64, 128, 256], "out_channels": 64})
    feats = [torch.randn(1, c, 32 // (2**i), 32 // (2**i)) for i, c in enumerate([64, 128, 256])]
    out = fpn(feats)
    assert len(out) == 3
    assert all(o.shape[1] == 64 for o in out)


def test_heads_shapes():
    cls_head = build_head({"type": "cls_head", "in_channels": 16, "num_classes": 5})
    assert cls_head(torch.randn(2, 16)).shape == (2, 5)

    seg_head = build_head(
        {"type": "seg_head", "in_channels": 16, "num_classes": 4, "output_size": [8, 8]}
    )
    assert seg_head(torch.randn(2, 16, 4, 4)).shape == (2, 4, 8, 8)

    det_head = build_head({"type": "det_head", "in_channels": 16, "num_classes": 3})
    out = det_head(torch.randn(2, 16, 8, 8))
    assert out.shape[1] == 5 + 3


def test_registries_contain_components():
    assert "fpn" in NECKS
    assert "cls_head" in HEADS
    assert "seg_head" in HEADS
    assert "det_head" in HEADS
    # 教学模型注册
    assert "SimpleCNN" in MODELS and "simple_cnn" in MODELS
    assert "GridDetectionModel" in MODELS


def test_toy_models_build_and_forward():
    cls_model = build_model({"type": "simple_cnn", "num_classes": 7})
    assert cls_model(torch.randn(2, 3, 32, 32)).shape == (2, 7)

    det_model = build_model({"type": "grid_detection", "num_classes": 3})
    out = det_model(torch.randn(2, 3, 64, 64))
    assert out.shape[1] == 5 + 3


def test_base_model_contract():
    from dvisionix.models.base import TASK_TYPES, BaseModel

    assert "classification" in TASK_TYPES
    with pytest.raises(ValueError):
        BaseModel(task_type="not_a_task")
