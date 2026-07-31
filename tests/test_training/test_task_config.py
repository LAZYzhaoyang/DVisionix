# -*- coding: utf-8 -*-
"""Task 组件化测试：配置驱动构建 / optimizer/scheduler/loss/metrics 注入。"""

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.training import (
    build_task,
    ClassificationTask,
    SegmentationTask,
    DetectionTask,
    OPTIMIZERS,
    SCHEDULERS,
    build_optimizer,
    build_scheduler,
)
from dvisionix.models import SimpleCNN
from dvisionix.models.losses import FocalLoss, CrossEntropy
from dvisionix.metrics import MetricCollection, get_preset_metrics


class _TinyDS(torch.utils.data.Dataset):
    def __init__(self, n=8, num_classes=3):
        self.n, self.nc = n, num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"image": torch.randn(3, 32, 32), "label": torch.randint(0, self.nc, ())}


def test_build_task_from_config_classification():
    task = build_task({
        "type": "ClassificationTask",
        "num_classes": 4,
        "optimizer_cfg": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01},
        "scheduler_cfg": {"type": "cosine", "T_max": 50},
        "loss": {"type": "focal", "gamma": 2.0},
    })
    assert isinstance(task, ClassificationTask)
    assert isinstance(task.loss, FocalLoss)
    opt_cfg = task.configure_optimizers(SimpleCNN(num_classes=4))
    assert isinstance(opt_cfg["optimizer"], torch.optim.AdamW)
    assert isinstance(opt_cfg["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
    assert task.metrics is not None and len(task.metrics) == 4  # acc/p/r/f1


def test_build_task_from_config_detection():
    task = build_task({"type": "DetectionTask", "num_classes": 3})
    assert isinstance(task, DetectionTask)
    assert task.metrics is not None


def test_build_task_from_config_segmentation():
    task = build_task({"type": "SegmentationTask", "num_classes": 5})
    assert isinstance(task, SegmentationTask)
    assert isinstance(task.loss, CrossEntropy)
    assert task.metrics is not None and len(task.metrics) == 2  # mIoU/pixel_acc


def test_custom_metrics_injection():
    metrics = MetricCollection([
        {"type": "accuracy"},
        {"type": "top_k_accuracy", "k": 2},
    ])
    task = ClassificationTask(num_classes=3, metrics=metrics)
    assert task.metrics is not None and len(task.metrics) == 2


def test_optimizer_and_scheduler_registries():
    assert "adamw" in OPTIMIZERS and "sgd" in OPTIMIZERS
    assert "cosine" in SCHEDULERS and "reduce_on_plateau" in SCHEDULERS
    model = SimpleCNN(num_classes=3)
    opt = build_optimizer({"type": "sgd", "lr": 0.1, "momentum": 0.9}, model.parameters())
    assert isinstance(opt, torch.optim.SGD)
    sched, monitor = build_scheduler({"type": "cosine", "T_max": 10}, opt)
    assert monitor is None
    sched2, monitor2 = build_scheduler({"type": "reduce_on_plateau", "monitor": "val_loss"}, opt)
    assert monitor2 == "val_loss"


def test_validation_metrics_integration():
    """Trainer 验证循环应产出任务指标（accuracy 等）。"""
    from dvisionix.training import Trainer
    from dvisionix.models import SimpleCNN

    train_ds = _TinyDS(n=8, num_classes=3)
    val_ds = _TinyDS(n=6, num_classes=3)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    task = ClassificationTask(num_classes=3)
    trainer = Trainer(task, train_loader, val_loader, max_epochs=1, device="cpu", log_interval=999)
    result = trainer.fit(SimpleCNN(num_classes=3))
    last_epoch = result["history"][-1]
    assert "train_loss" in last_epoch and "val_loss" in last_epoch
    assert "accuracy" in last_epoch  # MetricCollection 指标
    assert "history" in result and len(result["history"]) == 1