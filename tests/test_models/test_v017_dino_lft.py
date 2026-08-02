# -*- coding: utf-8 -*-
"""v0.17 DINO 可选增强：look-forward-twice（LFT）。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.models import build_model
from dvisionix.models.losses import DINOLoss

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BB = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}
HEAD = {
    "type": "dino_head",
    "num_classes": 3,
    "d_model": 64,
    "num_queries": 10,
    "num_encoder_layers": 1,
    "num_decoder_layers": 2,
    "num_heads": 4,
    "num_points": 2,
    "topk": 8,
}
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


def _build():
    return build_model({"type": "dinodetr", "num_classes": 3, "backbone": BB, "head": HEAD})


def test_dino_intermediate_boxes_train_only():
    model = _build()
    model.train()
    preds = model(BATCH["image"], batch=BATCH)
    assert "intermediate_boxes" in preds
    inter = preds["intermediate_boxes"]
    assert len(inter) == HEAD["num_decoder_layers"] == 2
    for boxes_i in inter:
        assert boxes_i.shape == (2, HEAD["topk"], 4)
    # 最后一层累积框 = 最终输出框
    torch.testing.assert_close(inter[-1], preds["boxes"], rtol=1e-6, atol=1e-6)

    # 推理路径不受影响：不输出中间框 / 去噪项
    model.eval()
    with torch.no_grad():
        preds_eval = model(BATCH["image"])
    assert "intermediate_boxes" not in preds_eval
    assert "dn_logits" not in preds_eval
    boxes, scores, labels = model.decode(preds_eval, (128, 128), score_threshold=0.0)
    assert len(boxes) == 2 and boxes[0].shape[1] == 4


def test_dino_lft_loss_decreases():
    torch.manual_seed(0)
    model = _build()
    loss_fn = DINOLoss(num_classes=3, lft=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    first = last = None
    for _ in range(8):
        opt.zero_grad()
        out = loss_fn(model(BATCH["image"], batch=BATCH), BATCH, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_dino_lft_layer_weights():
    loss_fn = DINOLoss(num_classes=3, layer_weights=[0.5, 1.0])
    model = _build()
    model.train()
    out = loss_fn(model(BATCH["image"], batch=BATCH), BATCH, image_hw=(128, 128))
    assert out["loss"].dim() == 0

    # 长度不一致应报错
    bad = DINOLoss(num_classes=3, layer_weights=[1.0])
    with pytest.raises(ValueError, match="layer_weights"):
        bad(model(BATCH["image"], batch=BATCH), BATCH, image_hw=(128, 128))


def test_dino_lft_fallback_when_disabled_or_missing():
    """lft=False 或无中间框时回退单层 DETRLoss，行为不破坏。"""
    model = _build()
    model.train()
    preds = model(BATCH["image"], batch=BATCH)
    loss_off = DINOLoss(num_classes=3, lft=False)
    out = loss_off(preds, BATCH, image_hw=(128, 128))
    assert out["loss"].dim() == 0

    # 无 intermediate_boxes（例如手工裁剪输出）时也正常
    preds_min = {k: v for k, v in preds.items() if k != "intermediate_boxes"}
    out2 = DINOLoss(num_classes=3)(preds_min, BATCH, image_hw=(128, 128))
    assert out2["loss"].dim() == 0
