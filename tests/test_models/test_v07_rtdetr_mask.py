# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: v0.7 测试：RT-DETR / Mask mAP 指标 / maskformer_decode / evaluate...
"""v0.7 测试：RT-DETR / Mask mAP 指标 / maskformer_decode / evaluate_mask_ap。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.metrics import MaskAveragePrecision
from dvisionix.models import build_model, maskformer_decode
from dvisionix.models.losses import DETRLoss
from dvisionix.registry import HEADS, METRICS, MODELS

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}


def test_rtdetr_registered_forward_decode():
    model = build_model(
        {
            "type": "rtdetr",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "rtdetr_head",
                "num_classes": 3,
                "d_model": 32,
                "topk": 8,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        }
    )
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"logits", "boxes"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


def test_rtdetr_loss_decreases():
    torch.manual_seed(0)
    model = build_model(
        {
            "type": "rtdetr",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "rtdetr_head",
                "num_classes": 3,
                "d_model": 32,
                "topk": 8,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        }
    )
    loss_fn = DETRLoss(num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    samples = [
        {
            "image": torch.randn(3, 128, 128),
            "boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0], [70.0, 70.0, 100.0, 100.0]]),
            "labels": torch.tensor([0, 1]),
        }
        for _ in range(2)
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


def test_mask_ap_perfect_and_wrong():
    h = w = 8
    m1 = torch.zeros(1, h, w, dtype=torch.bool)
    m1[0, 2:6, 2:6] = True
    m2 = torch.zeros(1, h, w, dtype=torch.bool)
    m2[0, 4:7, 4:7] = True

    perfect = MaskAveragePrecision(num_classes=2)
    perfect.update([m1], [torch.tensor([0.9])], [torch.tensor([0])], [m1], [torch.tensor([0])])
    assert perfect.compute()["mask_mAP_50"] > 0.99

    wrong = MaskAveragePrecision(num_classes=2)
    wrong.update(
        [torch.zeros(1, h, w, dtype=torch.bool)],
        [torch.tensor([0.9])],
        [torch.tensor([0])],
        [m1],
        [torch.tensor([0])],
    )
    assert wrong.compute()["mask_mAP_50"] == 0.0


def test_maskformer_decode_and_evaluate():
    from dvisionix.training import evaluate_mask_ap

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
    preds = model(torch.randn(1, 3, 64, 64))
    masks, scores, labels = maskformer_decode(preds, (64, 64), score_threshold=0.0)
    assert masks[0].ndim == 3 and masks[0].dtype == torch.bool

    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {
                "image": torch.randn(3, 64, 64),
                "mask": torch.randint(0, 3, (64, 64)),
                "labels": torch.tensor([0]),
            }

    loader = DataLoader(
        DS(),
        batch_size=2,
        collate_fn=lambda b: {
            "image": torch.stack([x["image"] for x in b]),
            "mask": torch.stack([x["mask"] for x in b]),
            "labels": [x["labels"] for x in b],
        },
    )
    result = evaluate_mask_ap(
        model, loader, num_classes=3, device=torch.device("cpu"), score_threshold=0.0
    )
    assert set(result.keys()) == {"mask_mAP", "mask_mAP_50", "mask_mAP_75"}
    assert "mask_ap" in METRICS and "rtdetr" in MODELS and "rtdetr_head" in HEADS
