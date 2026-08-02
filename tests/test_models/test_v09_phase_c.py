# -*- coding: utf-8 -*-
"""v0.9.0 阶段 C 测试：全景评估接入验证循环 / RT-DETR 增强版 / Mask2Former 完整版。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.models import build_model
from dvisionix.models.losses import DETRLoss, MaskFormerLoss
from dvisionix.registry import HEADS, MODELS
from dvisionix.training import MaskFormerTask

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}


# ---------- C1: 全景评估接入验证循环 ----------


def test_maskformer_task_panoptic_validation():
    assert "mask2former_head" in HEADS or "maskformer_head" in HEADS
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "maskformer_head",
                "num_classes": 3,
                "d_model": 32,
                "num_queries": 8,
                "output_mode": "full",
            },
        }
    )
    task = MaskFormerTask(num_classes=3, panoptic=True)
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 3, (2, 64, 64)),
        "labels": [torch.tensor([0]), torch.tensor([1])],
    }
    vr = task.validation_step(model, batch, torch.device("cpu"))
    # preds 扩展为 4 元组（含全景 id 图），targets 扩展为 3 元组
    assert len(vr["preds"]) == 4 and len(vr["targets"]) == 3
    assert vr["preds"][3][0].shape == (64, 64) and vr["preds"][3][0].dtype == torch.int64
    task.update_metrics(vr["preds"], vr["targets"])
    result = task.on_validation_epoch_end()
    assert set(result) >= {"mask_mAP", "PQ", "SQ", "RQ"}


def test_maskformer_task_panoptic_off_unchanged():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "maskformer_head",
                "num_classes": 3,
                "d_model": 32,
                "num_queries": 8,
                "output_mode": "full",
            },
        }
    )
    task = MaskFormerTask(num_classes=3)  # panoptic 默认 False
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 3, (2, 64, 64)),
        "labels": [torch.tensor([0]), torch.tensor([1])],
    }
    vr = task.validation_step(model, batch, torch.device("cpu"))
    assert len(vr["preds"]) == 3 and len(vr["targets"]) == 2


# ---------- C2: RT-DETR 增强版 ----------


def test_rtdetr_full_forward_decode_and_loss():
    assert "rtdetr_full" in MODELS and "rtdetr_full_head" in HEADS
    head = {
        "type": "rtdetr_full_head",
        "num_classes": 3,
        "d_model": 64,
        "topk": 8,
        "num_encoder_layers": 1,
        "num_decoder_layers": 2,
        "num_heads": 4,
        "num_points": 2,
    }
    model = build_model(
        {"type": "rtdetr_full", "num_classes": 3, "backbone": BACKBONE, "head": head}
    )
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"logits", "boxes"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4

    torch.manual_seed(0)
    loss_fn = DETRLoss(num_classes=3)
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
    for _ in range(5):
        opt.zero_grad()
        out = loss_fn(model(batch["image"]), batch, image_hw=(128, 128))
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


# ---------- C3: Mask2Former 完整版 ----------


def test_mask2former_forward_decode():
    assert "mask2former_head" in HEADS
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "mask2former_head",
                "num_classes": 3,
                "d_model": 32,
                "num_queries": 8,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        }
    )
    preds = model(torch.randn(1, 3, 64, 64))
    assert set(preds.keys()) == {"pred_logits", "pred_masks", "semantic_logits"}
    masks, scores, labels = model.decode(preds, (64, 64), score_threshold=0.0)
    assert masks[0].ndim == 3 and masks[0].dtype == torch.bool


def test_mask2former_loss_decreases_semantic_gt():
    torch.manual_seed(0)
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "mask2former_head",
                "num_classes": 3,
                "d_model": 32,
                "num_queries": 8,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        }
    )
    loss_fn = MaskFormerLoss(num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 3, (2, 64, 64)),
        "labels": [torch.tensor([0]), torch.tensor([1])],
    }
    first = last = None
    for _ in range(5):
        opt.zero_grad()
        out = loss_fn(model(batch["image"]), batch)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first


def test_maskformer_loss_instance_gt():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "mask2former_head",
                "num_classes": 3,
                "d_model": 32,
                "num_queries": 8,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        }
    )
    loss_fn = MaskFormerLoss(num_classes=3)
    batch = {
        "image": torch.randn(1, 3, 64, 64),
        "mask": torch.randint(0, 3, (1, 64, 64)),
        "instance_masks": [torch.tensor([[[1, 0], [0, 0]], [[0, 1], [1, 0]]]).float()],
        "instance_labels": [torch.tensor([0, 1])],
    }
    out = loss_fn(model(batch["image"]), batch)
    assert out["loss"].requires_grad
