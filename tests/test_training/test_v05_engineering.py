# -*- coding: utf-8 -*-
"""v0.5 工程能力测试：EMA / 蒸馏 / checkpoint 保留 / history.csv / MaskFormerLoss。"""

import os

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import SimpleCNN, build_model
from dvisionix.models.losses import DistillationLoss, MaskFormerLoss
from dvisionix.registry import LOSSES
from dvisionix.training import EMA, ClassificationTask, DistillCallback, ModelCheckpoint, Trainer

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]


class _DS(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"image": torch.randn(3, 32, 32), "label": torch.tensor(i % 3)}


def _loader():
    from torch.utils.data import DataLoader

    return DataLoader(_DS(), batch_size=4)


def test_ema_callback_trains():
    task = ClassificationTask(num_classes=3, learning_rate=1e-2)
    ema = EMA(decay=0.9)
    trainer = Trainer(
        task, _loader(), _loader(), callbacks=[ema], max_epochs=1, device="cpu", log_interval=999
    )
    trainer.fit(SimpleCNN(num_classes=3))
    assert len(ema.shadow) > 0


def test_history_csv_exported(tmp_path):
    task = ClassificationTask(num_classes=3, learning_rate=1e-2)
    trainer = Trainer(
        task,
        _loader(),
        _loader(),
        work_dir=str(tmp_path),
        max_epochs=1,
        device="cpu",
        log_interval=999,
    )
    trainer.fit(SimpleCNN(num_classes=3))
    csv_path = os.path.join(tmp_path, "history.csv")
    assert os.path.exists(csv_path)
    assert "train_loss" in open(csv_path, encoding="utf-8").read()


def test_checkpoint_retention(tmp_path):
    ckpt = ModelCheckpoint(
        save_dir=str(tmp_path),
        save_best_only=False,
        save_last=True,
        save_every_n_epochs=1,
        max_epoch_checkpoints=2,
    )
    task = ClassificationTask(num_classes=3, learning_rate=1e-2)
    trainer = Trainer(
        task, _loader(), _loader(), callbacks=[ckpt], max_epochs=4, device="cpu", log_interval=999
    )
    trainer.fit(SimpleCNN(num_classes=3))
    epoch_files = [f for f in os.listdir(tmp_path) if f.startswith("epoch=")]
    assert 2 <= len(epoch_files) <= 2  # max_epoch_checkpoints 上限


def test_distill_loss_and_callback():
    loss = DistillationLoss(alpha=0.5, temperature=4.0)
    student = torch.randn(4, 5)
    teacher = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))
    out = loss(student, labels, teacher_logits=teacher)
    assert torch.isfinite(out)
    assert "DistillationLoss" in LOSSES
    teacher_model = SimpleCNN(num_classes=3)
    cb = DistillCallback(teacher=teacher_model, temperature=4.0)
    assert cb.temperature == 4.0


def test_maskformer_full_and_loss():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": True},
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
    assert isinstance(preds, dict) and set(preds.keys()) == {
        "pred_logits",
        "pred_masks",
        "semantic_logits",
    }
    loss_fn = MaskFormerLoss(num_classes=3)
    batch = {"mask": torch.randint(0, 3, (1, 16, 16))}
    out = loss_fn(preds, batch)
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()
    assert "MaskFormerLoss" in LOSSES
