# -*- coding: utf-8 -*-
"""v0.12.0 测试：ViT/Swin 骨干 / YOLOv9-lite（PGI）。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.models import build_model
from dvisionix.models.layers import build_layer
from dvisionix.models.losses import YOLOv9Loss
from dvisionix.registry import BACKBONES, LAYERS, LOSSES, MODELS

VIT_BB = {"type": "vit_backbone", "embed_dim": 96, "depth": 2, "num_heads": 4}
SWIN_BB = {
    "type": "swin_backbone",
    "embed_dim": 32,
    "depths": (1, 1, 1, 1),
    "num_heads": (2, 2, 2, 2),
}


# ---------- ViT / Swin 骨干 ----------


@pytest.mark.parametrize("bb", [VIT_BB, SWIN_BB])
def test_transformer_backbones_cls_and_det(bb):
    assert bb["type"] in BACKBONES
    m = build_model(
        {
            "type": "linear_classifier",
            "backbone": bb,
            "head": {"type": "cls_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    out = m(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 5)

    strides = [8] if bb["type"] == "vit_backbone" else [8, 16, 32]
    m2 = build_model(
        {
            "type": "fcos",
            "num_classes": 3,
            "backbone": {**bb, "features_only": True},
            "neck": {"type": "fpn", "out_channels": 64},
            "head": {"type": "fcos_head", "num_classes": 3, "strides": strides},
        }
    )
    preds = m2(torch.randn(1, 3, 32, 32))
    boxes, scores, labels = m2.decode(preds, (32, 32), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


def test_vit_arbitrary_input_size():
    """正弦位置编码支持非训练尺寸输入。"""
    bb = build_model(
        {
            "type": "linear_classifier",
            "backbone": {**VIT_BB, "features_only": True},
            "head": {"type": "fcn_head", "num_classes": 4},
            "num_classes": 4,
        }
    )
    for size in (32, 48, 64):
        out = bb.backbone(torch.randn(1, 3, size, size))
        assert isinstance(out, list) and out[0].shape[1] == 96


# ---------- YOLOv9-lite ----------


def test_reversible_block_inverse():
    assert "reversible_block" in LAYERS
    rb = build_layer({"type": "reversible_block", "channels": 64, "num_layers": 2})
    x = torch.randn(2, 64, 16, 16)
    y = rb(x)
    xr = rb.inverse(y)
    assert (x - xr).abs().max().item() < 1e-4


def test_yolov9_forward_decode_and_loss():
    assert "yolo_v9" in MODELS and "yolo_v9_detection" in LOSSES
    bb = {
        "type": "sequential_backbone",
        "features_only": True,
        "stages": [
            {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 4},
            {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
            [
                {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
                {"type": "eelan_layer", "in_channels": 64, "out_channels": 64, "num_blocks": 2},
            ],
            [
                {"type": "conv_norm_act", "in_channels": 64, "out_channels": 128, "stride": 2},
                {"type": "reversible_block", "channels": 128, "num_layers": 2},
            ],
        ],
    }
    cfg = {
        "type": "yolo_v9",
        "num_classes": 3,
        "backbone": bb,
        "neck": {"type": "panet", "out_channels": 64},
        "head": {"type": "yolo_head", "num_classes": 3, "strides": [4, 8, 16, 32]},
        "aux_head": {"type": "yolo_head", "num_classes": 3, "strides": [4]},
        "aux_stage_index": 1,
    }
    model = build_model(cfg)
    model.train()
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds) >= {"cls", "reg", "aux_cls", "aux_reg"}
    model.eval()
    with torch.no_grad():
        preds_eval = model(torch.randn(1, 3, 128, 128))
        boxes, scores, labels = model.decode(preds_eval, (128, 128), score_threshold=0.0)
    assert "aux_cls" not in preds_eval
    assert len(boxes) == 1 and boxes[0].shape[1] == 4

    model.train()  # 训练模式启用 PGI 辅助头
    torch.manual_seed(0)
    loss_fn = YOLOv9Loss(num_classes=3, strides=(4, 8, 16, 32), aux_strides=(4,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    samples = [
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
    batch = detection_collate(samples)
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
    assert "aux_cls_loss" in out


def test_yolov9_config_loads():
    from dvisionix.config import Config

    cfg = Config.from_yaml(r"D:\ZhaoyangProject\DVisionix\configs\detection\yolov9_synthetic.yaml")
    m = build_model(cfg.model.to_dict())
    assert m is not None
