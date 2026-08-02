# -*- coding: utf-8 -*-
"""v0.8.0 阶段 B 测试：分割头 / MaskFormerTask / PanopticQuality / YOLO 系列 / Deformable DETR / 分类头。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.metrics import PanopticQuality
from dvisionix.models import build_model
from dvisionix.models.layers import build_layer
from dvisionix.models.losses import DETRLoss
from dvisionix.registry import HEADS, LAYERS, METRICS, MODELS, TASKS
from dvisionix.training import MaskFormerTask, panoptic_decode

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}


# ---------- B-分割：新分割头 ----------


@pytest.mark.parametrize(
    "head",
    [
        {"type": "psp_head", "num_classes": 4},
        {"type": "upernet_head", "num_classes": 4},
        {"type": "deeplabv3plus_head", "num_classes": 4},
    ],
)
def test_new_seg_heads_forward(head):
    for name in ("psp_head", "upernet_head", "deeplabv3plus_head"):
        assert name in HEADS, name
    model = build_model(
        {"type": "segmentation_model", "num_classes": 4, "backbone": BACKBONE, "head": head}
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


def test_upernet_multiscale_injection():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": BACKBONE,
            "neck": {"type": "fpn", "out_channels": 48},
            "head": {"type": "upernet_head", "num_classes": 4},
        }
    )
    assert model.head.in_channels_list == [48, 48, 48]


# ---------- B-分割：MaskFormerTask + panoptic ----------


def test_maskformer_task_training_and_validation():
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
    task = MaskFormerTask(num_classes=3)
    assert "MaskFormerTask" in TASKS
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 3, (2, 64, 64)),
        "labels": [torch.tensor([0]), torch.tensor([1])],
    }
    r = task.training_step(model, batch, torch.device("cpu"))
    assert r["loss"].requires_grad and set(r) >= {"cls_loss", "mask_bce_loss", "mask_dice_loss"}
    vr = task.validation_step(model, batch, torch.device("cpu"))
    assert vr["preds"][0][0].ndim == 3 and vr["preds"][0][0].dtype == torch.bool


def test_panoptic_quality_perfect_and_error():
    gt = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    m = PanopticQuality(num_categories=3)
    m.update(gt.clone(), gt.clone())
    perfect = m.compute()
    assert perfect["PQ"] == pytest.approx(1.0) and perfect["RQ"] == pytest.approx(1.0)
    m2 = PanopticQuality(num_categories=3)
    m2.update(gt + 2000, gt.clone())
    assert m2.compute()["PQ"] == pytest.approx(0.0)
    assert "panoptic_quality" in METRICS


def test_panoptic_decode_shape():
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
    preds = model(torch.randn(2, 3, 64, 64))
    pans = panoptic_decode(preds, (64, 64), num_classes=3, score_threshold=0.0)
    assert len(pans) == 2 and pans[0].shape == (64, 64) and pans[0].dtype == torch.int64


# ---------- B-检测：YOLO 系列 ----------


@pytest.mark.parametrize("layer", ["csp_layer", "elan_layer"])
def test_yolo_layers(layer):
    assert layer in LAYERS
    out = build_layer({"type": layer, "in_channels": 32, "out_channels": 64, "num_blocks": 2})(
        torch.randn(1, 32, 16, 16)
    )
    assert out.shape == (1, 64, 16, 16)


def test_yolov5_style_detector():
    stages = [
        {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 4},
        {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
        [
            {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
            {"type": "csp_layer", "in_channels": 64, "out_channels": 64, "num_blocks": 2},
        ],
        [
            {"type": "conv_norm_act", "in_channels": 64, "out_channels": 128, "stride": 2},
            {"type": "csp_layer", "in_channels": 128, "out_channels": 128, "num_blocks": 2},
        ],
    ]
    model = build_model(
        {
            "type": "yolo",
            "num_classes": 3,
            "backbone": {"type": "sequential_backbone", "stages": stages, "features_only": True},
            "neck": {"type": "panet", "out_channels": 64},
            "head": {"type": "yolo_head", "num_classes": 3, "strides": [4, 8, 16, 32]},
        }
    )
    preds = model(torch.randn(1, 3, 128, 128))
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


# ---------- B-检测：Deformable DETR ----------


def test_deformable_detr_forward_decode_and_loss():
    assert "deformable_detr" in MODELS and "deformable_detr_head" in HEADS
    model = build_model(
        {
            "type": "deformable_detr",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "deformable_detr_head",
                "num_classes": 3,
                "d_model": 64,
                "num_queries": 10,
                "num_encoder_layers": 1,
                "num_decoder_layers": 2,
                "num_heads": 4,
                "num_points": 2,
            },
        }
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


# ---------- B-分类：新度量头 ----------


def test_normface_and_curricularface():
    for name in ("normface_head", "curricularface_head"):
        assert name in HEADS
        model = build_model(
            {
                "type": "linear_classifier",
                "backbone": {
                    "type": "sequential_backbone",
                    "stages": STAGES,
                    "features_only": False,
                },
                "head": {"type": name, "num_classes": 5},
                "num_classes": 5,
            }
        )
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 5)


def test_partial_fc_full_and_sampling():
    model = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "partial_fc_head", "num_classes": 100},
            "num_classes": 100,
        }
    )
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 100)

    head = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "partial_fc_head", "num_classes": 100, "num_sample_classes": 16},
            "num_classes": 100,
        }
    )
    head.train()
    feat = head.backbone(torch.randn(4, 3, 64, 64))
    labels = torch.tensor([3, 3, 42, 7])
    logits, sampled = head.head(feat, labels)
    assert logits.shape[0] == 4 and sampled.numel() == 16
    assert set(labels.tolist()) <= set(sampled.tolist())
    local = head.head.remap_labels(labels, sampled)
    assert local.shape == (4,)
