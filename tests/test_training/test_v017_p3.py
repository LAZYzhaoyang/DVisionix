# -*- coding: utf-8 -*-
"""v0.17-P3 测试：性能开关（torch.compile / channels_last）+ 实验管理（配置哈希 / best_metrics / ONNX 导出）。"""

import csv
import os

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.models import SimpleCNN
from dvisionix.training import (
    ClassificationTask,
    ModelCheckpoint,
    Trainer,
    build_trainer,
    hash_config,
    resolve_work_dir,
)


class _TinyDS(torch.utils.data.Dataset):
    def __init__(self, n=8, num_classes=3):
        self.n, self.nc = n, num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"image": torch.randn(3, 32, 32), "label": torch.randint(0, self.nc, ())}


def _make_loader(n=8, bs=2):
    return DataLoader(_TinyDS(n=n), batch_size=bs)


def _make_trainer(**kwargs):
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("log_interval", 999)
    return Trainer(
        ClassificationTask(num_classes=3, learning_rate=1e-2),
        _make_loader(),
        _make_loader(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 实验管理：配置哈希 + work_dir 后缀
# ---------------------------------------------------------------------------
def test_hash_config_deterministic():
    cfg_a = Config({"experiment_name": "exp", "training": {"lr": 0.001, "opt": {"type": "adam"}}})
    cfg_b = Config({"experiment_name": "exp", "training": {"lr": 0.001, "opt": {"type": "adam"}}})
    cfg_c = Config({"experiment_name": "exp", "training": {"lr": 0.01, "opt": {"type": "adam"}}})
    assert hash_config(cfg_a) == hash_config(cfg_b)
    assert hash_config(cfg_a) != hash_config(cfg_c)
    assert len(hash_config(cfg_a)) == 8


def test_resolve_work_dir_hash_suffix(tmp_path):
    cfg = Config({"experiment_name": "exp_c"})
    wd = resolve_work_dir(cfg, cli_work_dir=str(tmp_path))
    tail = os.path.basename(wd)
    # 格式：<timestamp>-<hash8>
    assert "-" in tail and len(tail.split("-")[-1]) == 8


# ---------------------------------------------------------------------------
# 性能开关：channels_last / torch.compile（失败降级）
# ---------------------------------------------------------------------------
def test_trainer_channels_last():
    trainer = _make_trainer(channels_last=True)
    model = SimpleCNN(num_classes=3, in_channels=3)
    trainer.fit(model)
    # 卷积参数应为 channels_last 内存格式
    conv = next(m for m in trainer.model.modules() if isinstance(m, torch.nn.Conv2d))
    assert conv.weight.is_contiguous(memory_format=torch.channels_last)


def test_trainer_compile_fallback(monkeypatch):
    """torch.compile 抛错时应告警降级，训练照常进行。"""

    def boom(*args, **kwargs):
        raise RuntimeError("compile not supported")

    monkeypatch.setattr(torch, "compile", boom)
    trainer = _make_trainer(compile=True)
    model = SimpleCNN(num_classes=3, in_channels=3)
    trainer.fit(model)
    assert isinstance(trainer.model, SimpleCNN)


def test_builder_wires_perf_switches(tmp_path):
    from dvisionix.training import build_callbacks

    cfg = Config(
        {
            "task_type": "classification",
            "experiment_name": "exp_d",
            "model": {"name": "simple_cnn", "num_classes": 3, "in_channels": 3},
            "data": {"image_size": 32},
            "training": {
                "num_epochs": 1,
                "device": "cpu",
                "compile": True,
                "channels_last": True,
                "optimizer": {"type": "adam", "lr": 1e-3},
            },
            "checkpoint": {"save_dir": "checkpoints"},
        }
    )
    cbs = build_callbacks(cfg, work_dir=str(tmp_path))
    trainer = build_trainer(
        cfg,
        ClassificationTask(num_classes=3, learning_rate=1e-3),
        _make_loader(),
        _make_loader(),
        work_dir=str(tmp_path),
    )
    assert trainer.compile is True
    assert trainer.channels_last is True
    assert len(cbs) >= 1


# ---------------------------------------------------------------------------
# 实验管理：best_metrics.csv
# ---------------------------------------------------------------------------
def test_best_metrics_csv(tmp_path):
    ckpt_dir = os.path.join(tmp_path, "checkpoints")
    callbacks = [
        ModelCheckpoint(save_dir=ckpt_dir, monitor="val_loss", mode="min", save_best_only=True)
    ]
    trainer = _make_trainer(callbacks=callbacks, max_epochs=2, work_dir=str(tmp_path), seed=0)
    trainer.fit(SimpleCNN(num_classes=3, in_channels=3))
    path = os.path.join(tmp_path, "best_metrics.csv")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    # best_epoch 应指向 history 中 val_loss 最小的 epoch
    best_idx = int(row["best_epoch"])
    vals = [float(h["val_loss"]) for h in trainer.history]
    assert best_idx == vals.index(min(vals))


# ---------------------------------------------------------------------------
# 实验管理：导出最优 checkpoint 为 ONNX
# ---------------------------------------------------------------------------
def test_export_best_onnx(tmp_path):
    onnx = pytest.importorskip("onnx")
    del onnx
    from tools import train as train_tool

    cfg = Config(
        {
            "task_type": "classification",
            "model": {"name": "simple_cnn", "num_classes": 3, "in_channels": 3},
            "data": {"image_size": 32},
        }
    )
    ckpt_dir = os.path.join(tmp_path, "checkpoints")
    os.makedirs(ckpt_dir)
    model = SimpleCNN(num_classes=3, in_channels=3)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(ckpt_dir, "best.pt"))

    out = train_tool.export_best_onnx(cfg, str(tmp_path))
    assert out and os.path.exists(out)
    assert out.endswith("best.onnx")


def test_export_best_onnx_missing_ckpt(tmp_path):
    from tools import train as train_tool

    cfg = Config(
        {
            "task_type": "classification",
            "model": {"name": "simple_cnn", "num_classes": 3, "in_channels": 3},
            "data": {"image_size": 32},
        }
    )
    assert train_tool.export_best_onnx(cfg, str(tmp_path)) is None
