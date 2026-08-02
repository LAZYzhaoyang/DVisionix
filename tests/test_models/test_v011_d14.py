# -*- coding: utf-8 -*-
"""v0.11.0 测试：内置骨干 / SegFormerV2+SwinUNet / SimCLR 端到端。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data.transforms import SimCLRTransforms
from dvisionix.models import build_model
from dvisionix.registry import BACKBONES, HEADS, TASKS, TRANSFORMS
from dvisionix.training import SimCLRTask

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
SEQ_BB = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}


# ---------- D-1：内置骨干 ----------


@pytest.mark.parametrize(
    "name", ["convnext_backbone", "cspdarknet_backbone", "mobilenetv3_backbone"]
)
def test_builtin_backbones_cls_and_det(name):
    assert name in BACKBONES
    # 分类
    m = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": name, "features_only": False},
            "head": {"type": "cls_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    out = m(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 5)
    # 检测组合（features_only + FCOS）
    m2 = build_model(
        {
            "type": "fcos",
            "num_classes": 3,
            "backbone": {"type": name, "features_only": True},
            "neck": {"type": "fpn", "out_channels": 64},
            "head": {"type": "fcos_head", "num_classes": 3, "strides": [8, 16, 32]},
        }
    )
    preds = m2(torch.randn(1, 3, 64, 64))
    boxes, scores, labels = m2.decode(preds, (64, 64), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


# ---------- D-4：SegFormerV2 / SwinUNet ----------


@pytest.mark.parametrize(
    "head",
    [
        {"type": "segformer_v2_head", "num_classes": 4, "d_model": 32},
        {"type": "swin_unet_decoder", "num_classes": 4, "d_model": 32},
    ],
)
def test_segformer_v2_and_swin_unet(head):
    assert head["type"] in HEADS
    model = build_model(
        {"type": "segmentation_model", "num_classes": 4, "backbone": SEQ_BB, "head": head}
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


# ---------- D-2：SimCLR 端到端 ----------


def test_simclr_transforms_dual_view():
    assert "simclr_transforms" in TRANSFORMS
    import numpy as np

    tf = SimCLRTransforms(image_size=32, train=True)
    out = tf({"image": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "label": 3})
    assert out["image1"].shape == (3, 32, 32) and out["image2"].shape == (3, 32, 32)
    assert out["image1"].dtype == torch.float32 and out["label"] == 3
    assert not torch.allclose(out["image1"], out["image2"])


def test_simclr_task_training():
    assert "SimCLRTask" in TASKS
    torch.manual_seed(0)
    model = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "simclr_head", "num_classes": 5, "out_dim": 16},
            "num_classes": 5,
        }
    )
    task = SimCLRTask()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for _ in range(10):
        opt.zero_grad()
        # 正对：同一图像的两个视图（图像2 = 图像1 + 小噪声）
        x = torch.randn(8, 3, 32, 32)
        batch = {"image1": x, "image2": x + 0.1 * torch.randn_like(x)}
        r = task.training_step(model, batch, torch.device("cpu"))
        r["loss"].backward()
        opt.step()
        if first is None:
            first = r["loss"].item()
        last = r["loss"].item()
    assert last < first


def test_simclr_config_loads():
    from dvisionix.config import Config
    from dvisionix.training import build_task

    cfg = Config.from_yaml(
        r"D:\ZhaoyangProject\DVisionix\configs\classification\simclr_synthetic.yaml"
    )
    assert cfg.task_type == "simclr"
    model = build_model(cfg.model.to_dict())
    assert model is not None
    task = build_task({"type": "SimCLRTask", "num_classes": 16})
    assert isinstance(task, SimCLRTask)
