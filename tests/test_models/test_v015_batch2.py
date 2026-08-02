# -*- coding: utf-8 -*-
"""v0.15.0 批次 2 测试：SwinV2 / 可变形注意力 V2 / SegFormerV3 / 组合器子包。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import build_model
from dvisionix.models.layers import build_layer
from dvisionix.registry import BACKBONES, HEADS, LAYERS, MODELS

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
SEQ_BB = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}


# ---------- SwinV2 ----------


def test_rpb_and_swinv2_block():
    assert "continuous_relative_position_bias" in LAYERS and "swinv2_block" in LAYERS
    rpb = build_layer({"type": "continuous_relative_position_bias", "num_heads": 4})
    bias = rpb(4)
    assert bias.shape == (4, 16, 16)
    blk = build_layer({"type": "swinv2_block", "dim": 32, "num_heads": 4, "window_size": 4})
    out = blk(torch.randn(2, 32, 16, 16))
    assert out.shape == (2, 32, 16, 16)


def test_swinv2_backbone_cls_and_det():
    assert "swinv2_backbone" in BACKBONES
    kw = {"embed_dim": 32, "depths": (1, 1, 1, 1), "num_heads": (2, 2, 2, 2)}
    m = build_model(
        {
            "type": "linear_classifier",
            "backbone": {**kw, "type": "swinv2_backbone", "features_only": False},
            "head": {"type": "cls_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 5)
    m2 = build_model(
        {
            "type": "fcos",
            "num_classes": 3,
            "backbone": {**kw, "type": "swinv2_backbone", "features_only": True},
            "neck": {"type": "fpn", "out_channels": 64},
            "head": {"type": "fcos_head", "num_classes": 3, "strides": [8, 16, 32]},
        }
    )
    m2.eval()
    with torch.no_grad():
        preds = m2(torch.randn(1, 3, 32, 32))
        boxes, scores, labels = m2.decode(preds, (32, 32), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


# ---------- 可变形注意力 V2 ----------


def test_deformable_attention_v2():
    assert "multi_scale_deformable_attention_v2" in LAYERS
    attn = build_layer(
        {
            "type": "multi_scale_deformable_attention_v2",
            "embed_dim": 16,
            "num_heads": 4,
            "num_levels": 3,
            "num_points": 4,
        }
    )
    value_list = [torch.randn(2, 16, 16, 16), torch.randn(2, 16, 8, 8), torch.randn(2, 16, 4, 4)]
    q = torch.randn(2, 8, 16)
    ref = torch.rand(2, 8, 2)
    out = attn(q, value_list, ref)
    assert out.shape == (2, 8, 16)
    out.sum().backward()
    assert attn.level_offset.grad is not None


# ---------- SegFormerV3 ----------


def test_segformer_v3_head():
    assert "segformer_v3_head" in HEADS
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": SEQ_BB,
            "head": {"type": "segformer_v3_head", "num_classes": 4, "d_model": 32},
        }
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


# ---------- 组合器子包 ----------


def test_combinator_subpackages():
    assert (
        "linear_classifier" in MODELS and "segmentation_model" in MODELS and "swin_unet" in MODELS
    )
    m = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "cls_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    out = m(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 5)
    s = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": SEQ_BB,
            "head": {"type": "fcn_head", "num_classes": 4},
        }
    )
    assert s is not None
    u = build_model(
        {
            "type": "swin_unet",
            "num_classes": 4,
            "backbone": {
                "type": "swin_backbone",
                "embed_dim": 32,
                "depths": (1, 1, 1, 1),
                "num_heads": (2, 2, 2, 2),
            },
            "d_model": 32,
        }
    )
    assert u is not None
