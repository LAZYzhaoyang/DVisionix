# -*- coding: utf-8 -*-
"""v0.16.0 测试：DINO-lite / 线性评估 / 训练工程 P1。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.data import detection_collate
from dvisionix.models import build_model
from dvisionix.models.losses import DINOLoss
from dvisionix.registry import HEADS, LAYERS, LOSSES, MODELS, TASKS

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


# ---------- DINO-lite ----------


def test_dino_forward_decode_and_loss():
    assert "dinodetr" in MODELS and "dino_head" in HEADS and "dino_detection" in LOSSES
    assert "query_selection" in LAYERS and "denoising_query_generator" in LAYERS
    model = build_model({"type": "dinodetr", "num_classes": 3, "backbone": BB, "head": HEAD})
    model.train()
    preds = model(BATCH["image"], batch=BATCH)
    assert set(preds) >= {"logits", "boxes", "dn_logits", "dn_boxes", "dn_positive_mask"}
    model.eval()
    with torch.no_grad():
        preds_eval = model(BATCH["image"])
        boxes, scores, labels = model.decode(preds_eval, (128, 128), score_threshold=0.0)
    assert "dn_logits" not in preds_eval
    assert len(boxes) == 2 and boxes[0].shape[1] == 4

    torch.manual_seed(0)
    model = build_model({"type": "dinodetr", "num_classes": 3, "backbone": BB, "head": HEAD})
    loss_fn = DINOLoss(num_classes=3)
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


def test_dino_config_loads():
    from dvisionix.config import Config

    cfg = Config.from_yaml(r"D:\ZhaoyangProject\DVisionix\configs\detection\dino_synthetic.yaml")
    assert build_model(cfg.model.to_dict()) is not None


# ---------- 训练工程 P1 ----------


def test_warmup_scheduler():
    import torch as _t

    from dvisionix.training.optim import build_scheduler

    opt = _t.optim.SGD([_t.nn.Parameter(_t.zeros(2))], lr=0.1)
    sched, monitor = build_scheduler(
        {
            "type": "linear_warmup",
            "warmup_epochs": 3,
            "scheduler": {"type": "step", "step_size": 5},
        },
        opt,
    )
    sched.step()
    assert monitor is None
    assert opt.param_groups[0]["lr"] > 0.01  # warmup 初期 lr 增大


def test_ema_enhancement():
    from dvisionix.training import EMA

    ema = EMA(decay=0.9, decay_warmup_epochs=2, save_final=True)
    assert ema.decay_warmup_epochs == 2 and ema.save_final is True


# ---------- 线性评估 ----------


def test_linear_eval_task():
    assert "LinearEvalTask" in TASKS
    from dvisionix.training import LinearEvalTask

    task = LinearEvalTask(num_classes=5)
    assert task.feature_norm is True


def test_linear_eval_end_to_end():
    from dvisionix.training import LinearEvalTask

    torch.manual_seed(0)
    model = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "cls_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    task = LinearEvalTask(num_classes=5)
    opt_cfg = task.configure_optimizers(model)
    opt = opt_cfg["optimizer"]
    assert task.linear is not None
    # 冻结 backbone：所有 backbone 参数 requires_grad=False
    assert all(not p.requires_grad for p in model.backbone.parameters())
    batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 5, (4,))}
    first = last = None
    for _ in range(5):
        opt.zero_grad()
        r = task.training_step(model, batch, torch.device("cpu"))
        r["loss"].backward()
        opt.step()
        if first is None:
            first = r["loss"].item()
        last = r["loss"].item()
    assert last < first
