# -*- coding: utf-8 -*-
"""v0.14.0 中期批次 1 测试：ConvNeXtV2 / EfficientNetLite / MiT / SwinUNet / YOLOv11(C3k2+PSA)。"""

import os

import pytest

torch = pytest.importorskip("torch")

# 测试配置根目录（仓库根 configs/，避免硬编码绝对路径）
CFG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs"
)

from dvisionix.data import detection_collate
from dvisionix.models import build_model
from dvisionix.models.layers import build_layer
from dvisionix.models.losses import CrossEntropy, YOLOLoss
from dvisionix.registry import BACKBONES, LAYERS, MODELS

BATCH = detection_collate(
    [
        {
            "image": torch.randn(3, 128, 128),
            "boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
        },
        {
            "image": torch.randn(3, 128, 128),
            "boxes": torch.tensor([[70.0, 70.0, 100.0, 100.0]]),
            "labels": torch.tensor([1]),
        },
    ]
)


# ---------- 新 layers ----------


@pytest.mark.parametrize(
    "name,cfg,in_shape,out_shape",
    [
        ("grn", {"type": "grn", "dim": 32, "channels_first": True}, (2, 32, 8, 8), (2, 32, 8, 8)),
        ("convnextv2_block", {"type": "convnextv2_block", "dim": 32}, (2, 32, 8, 8), (2, 32, 8, 8)),
        (
            "c3k2_block",
            {"type": "c3k2_block", "in_channels": 32, "out_channels": 64},
            (2, 32, 8, 8),
            (2, 64, 8, 8),
        ),
        ("psa_block", {"type": "psa_block", "in_channels": 64}, (2, 64, 8, 8), (2, 64, 8, 8)),
    ],
)
def test_new_layers(name, cfg, in_shape, out_shape):
    assert name in LAYERS
    layer = build_layer(cfg)
    out = layer(torch.randn(*in_shape))
    assert out.shape == out_shape


# ---------- 新骨干 ----------


@pytest.mark.parametrize(
    "name,kw",
    [
        ("convnextv2_backbone", {"depths": (1, 1, 1, 1), "dims": (32, 64, 128, 256)}),
        ("efficientnet_lite_backbone", {"width_mult": 0.5}),
        ("mit_backbone", {"embed_dims": (32, 64, 128, 256), "depths": (1, 1, 1, 1)}),
    ],
)
def test_new_backbones_cls_and_det(name, kw):
    assert name in BACKBONES
    m = build_model(
        {
            "type": "linear_classifier",
            "backbone": {**kw, "type": name, "features_only": False},
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
            "backbone": {**kw, "type": name, "features_only": True},
            "neck": {"type": "fpn", "out_channels": 64},
            "head": {"type": "fcos_head", "num_classes": 3, "strides": [4, 8, 16, 32]},
        }
    )
    m2.eval()
    with torch.no_grad():
        preds = m2(torch.randn(1, 3, 32, 32))
    boxes, scores, labels = m2.decode(preds, (32, 32), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


# ---------- SwinUNet ----------


def test_swin_unet_forward_and_loss():
    assert "swin_unet" in MODELS
    torch.manual_seed(0)
    model = build_model(
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
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 4, 32, 32)
    loss_fn = CrossEntropy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randint(0, 4, (1, 32, 32))
    first = last = None
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(torch.randn(1, 3, 32, 32)), target)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first


# ---------- YOLOv11 风格 ----------


def test_yolov11_style_assembly():
    bb = {
        "type": "sequential_backbone",
        "features_only": True,
        "stages": [
            {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 4},
            [
                {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
                {"type": "c3k2_block", "in_channels": 32, "out_channels": 32, "num_blocks": 2},
            ],
            [
                {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
                {"type": "c3k2_block", "in_channels": 64, "out_channels": 64, "num_blocks": 2},
            ],
            [
                {"type": "conv_norm_act", "in_channels": 64, "out_channels": 128, "stride": 2},
                {"type": "psa_block", "in_channels": 128},
            ],
        ],
    }
    model = build_model(
        {
            "type": "yolo",
            "num_classes": 3,
            "backbone": bb,
            "neck": {"type": "panet", "out_channels": 64},
            "head": {"type": "yolo_head", "num_classes": 3, "strides": [4, 8, 16, 32]},
        }
    )
    preds = model(torch.randn(1, 3, 128, 128))
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4

    torch.manual_seed(0)
    loss_fn = YOLOLoss(num_classes=3, strides=(4, 8, 16, 32))
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


def test_midterm_configs_load():
    from dvisionix.config import Config

    for name in ("yolov11_synthetic",):
        cfg = Config.from_yaml(os.path.join(CFG_ROOT, "detection", f"{name}.yaml"))
        assert build_model(cfg.model.to_dict()) is not None
    for name in ("swin_unet_synthetic", "segformer_mit_synthetic"):
        cfg = Config.from_yaml(os.path.join(CFG_ROOT, "segmentation", f"{name}.yaml"))
        assert build_model(cfg.model.to_dict()) is not None
