# -*- coding: utf-8 -*-
"""工作目录隔离与自动断点续训测试。"""

import os

import torch
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.models import SimpleCNN
from dvisionix.training import (
    ClassificationTask,
    ModelCheckpoint,
    Trainer,
    dump_config,
    find_checkpoint,
    find_latest_run,
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


def test_resolve_work_dir_defaults_outside_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("DVISIONIX_WORK_DIR", raising=False)
    monkeypatch.setattr("dvisionix.training.workdir.os.path.expanduser", lambda p: str(tmp_path))
    cfg = Config({"experiment_name": "exp_a"})
    wd = resolve_work_dir(cfg)
    assert str(tmp_path) in wd
    assert "exp_a" in wd
    assert not wd.endswith("DVisionix")  # 不落在代码库内


def test_resolve_work_dir_priority(tmp_path):
    cfg = Config({"experiment_name": "exp_b"})
    wd = resolve_work_dir(cfg, cli_work_dir=str(tmp_path))
    assert wd.startswith(str(tmp_path))


def test_find_latest_run_and_checkpoint(tmp_path):
    run_dir = os.path.join(tmp_path, "exp", "20260101-000000")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir)
    open(os.path.join(ckpt_dir, "last.pt"), "w").close()

    latest = find_latest_run(os.path.join(tmp_path, "exp"))
    assert latest == run_dir
    assert find_checkpoint(run_dir, "auto") == os.path.join(ckpt_dir, "last.pt")
    assert find_checkpoint(run_dir, False) is None
    assert find_checkpoint(run_dir, "/some/ckpt.pt") == "/some/ckpt.pt"
    assert find_latest_run(os.path.join(tmp_path, "missing")) is None


def test_dump_config(tmp_path):
    cfg = Config({"a": {"b": 1}, "c": [1, 2]})
    path = os.path.join(tmp_path, "config.resolved.yaml")
    dump_config(cfg, path)
    assert os.path.exists(path)
    loaded = Config.from_yaml(path)
    assert loaded.a.b == 1


def test_auto_resume_roundtrip(tmp_path):
    """训练 1 epoch 后保存 last.pt，重建 Trainer 自动续训应从 epoch 1 开始。"""
    torch.manual_seed(0)
    ckpt_dir = os.path.join(tmp_path, "checkpoints")
    task = ClassificationTask(num_classes=3, learning_rate=1e-2)
    callbacks = [
        ModelCheckpoint(
            save_dir=ckpt_dir, monitor="val_loss", mode="min", save_best_only=False, save_last=True
        )
    ]

    t1 = Trainer(
        task,
        _make_loader(),
        _make_loader(),
        callbacks=callbacks,
        max_epochs=1,
        device="cpu",
        log_interval=999,
        seed=42,
    )
    model1 = SimpleCNN(num_classes=3, in_channels=3)
    t1.fit(model1)
    last_pt = os.path.join(ckpt_dir, "last.pt")
    assert os.path.exists(last_pt)

    # 新 trainer 从 last.pt 续训：current_epoch 应变为 1，可继续训练
    t2 = Trainer(
        ClassificationTask(num_classes=3, learning_rate=1e-2),
        _make_loader(),
        _make_loader(),
        max_epochs=3,
        device="cpu",
        log_interval=999,
        resume_from=last_pt,
        seed=42,
    )
    model2 = SimpleCNN(num_classes=3, in_channels=3)
    result = t2.fit(model2)
    assert t2.current_epoch == 2  # 续训完成后停在最后一个 epoch（epoch 1、2）
    assert len(result["history"]) == 2  # epoch 1,2
    assert t2.global_step >= t1.global_step
