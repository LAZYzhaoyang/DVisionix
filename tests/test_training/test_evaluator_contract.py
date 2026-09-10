# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 统一 Evaluator 与梯度累积窗口的契约测试。
"""统一 Evaluator 与梯度累积窗口的契约测试（CodePlan 7.5 步骤 3-2 / 3-3）。

覆盖两处 v1.0.0 的问题：

- **指标路径分裂**：训练中验证与独立 ``validate()`` 是两套实现，且 ``validate()``
  完全不触发 ``on_validation_begin/end`` —— EMA 回调正是在这两个钩子里交换/恢复
  权重，所以独立验证报的是**未交换 EMA 权重**的指标。
- **梯度累积尾窗分母**：一律除以 ``accumulate_grad_batches``，
  末尾不足一个完整窗口时梯度被系统性低估。
"""

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.models import SimpleCNN
from dvisionix.training import ClassificationTask, Trainer
from dvisionix.training.callbacks import Callback
from dvisionix.training.trainer import _infer_batch_size

NUM_CLASSES = 3


class _DS(torch.utils.data.Dataset):
    """确定性数据集：图像固定，保证多次验证遍历看到完全相同的数据。"""

    def __init__(self, n: int = 4):
        gen = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 32, 32, generator=gen)
        self.labels = torch.arange(n) % NUM_CLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"image": self.images[i], "label": self.labels[i]}


def _trainer(n=4, batch_size=1, accum=1, callbacks=None):
    loader = DataLoader(_DS(n), batch_size=batch_size)
    return Trainer(
        ClassificationTask(num_classes=NUM_CLASSES),
        loader,
        loader,
        callbacks=callbacks,
        max_epochs=1,
        seed=0,
        accumulate_grad_batches=accum,
        log_interval=999,
    )


class _Recorder(Callback):
    """记录验证钩子调用（EMA 的权重交换就挂在这两个钩子上）。"""

    def __init__(self):
        self.events = []

    def on_validation_begin(self, trainer):
        self.events.append("begin")

    def on_validation_end(self, trainer):
        self.events.append("end")


@pytest.mark.unit
@pytest.mark.parametrize(
    "total,accum,expected",
    [
        (4, 1, [1, 1, 1, 1]),
        (4, 2, [2, 2, 2, 2]),  # 完整窗口：最后一个 batch 仍应除以 2
        (5, 2, [2, 2, 2, 2, 1]),  # 尾窗只有 1 个 batch
        (3, 2, [2, 2, 1]),
        (6, 3, [3, 3, 3, 3, 3, 3]),
        (7, 3, [3, 3, 3, 3, 3, 3, 1]),
        (2, 8, [2, 2]),  # 窗口比整个 epoch 还长
    ],
)
def test_accumulation_divisor_is_window_length(total, accum, expected):
    trainer = _trainer(accum=accum)
    got = [trainer._accumulation_divisor(i, total) for i in range(total)]
    assert got == expected


@pytest.mark.unit
def test_infer_batch_size_uses_explicit_image_keys():
    assert _infer_batch_size({"image": torch.zeros(4, 3, 8, 8)}) == 4
    # SimCLR：batch 同时含 image(路径堆叠) 与 image1/image2
    assert _infer_batch_size({"image1": torch.zeros(2, 3, 8, 8)}) == 2
    assert (
        _infer_batch_size({"image": torch.zeros(4, 3, 8, 8), "image1": torch.zeros(4, 3, 8, 8)})
        == 4
    )


@pytest.mark.unit
def test_infer_batch_size_raises_without_image():
    """取不到图像键时必须报错，不得静默按 1 计数。"""
    with pytest.raises(ValueError, match="batch size"):
        _infer_batch_size({"foo": torch.zeros(3)})


@pytest.mark.integration
def test_validate_uses_same_implementation_as_training_validation():
    """独立 validate() 必须与训练中验证给出相同的指标。"""
    trainer = _trainer(n=8, batch_size=4)
    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)

    history = trainer.fit(model)["history"][-1]
    fresh = trainer.validate(model)

    assert "val_accuracy" in history, sorted(history)
    assert fresh["accuracy"] == pytest.approx(history["val_accuracy"], abs=1e-6)
    assert fresh["loss"] == pytest.approx(history["val_loss"], abs=1e-6)


@pytest.mark.integration
def test_validate_fires_validation_callbacks():
    """独立 validate() 必须触发 on_validation_begin/end —— EMA 的权重交换依赖它们。"""
    recorder = _Recorder()
    trainer = _trainer(n=8, batch_size=4, callbacks=[recorder])
    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)

    trainer.validate(model)
    assert recorder.events == ["begin", "end"]

    recorder.events.clear()
    trainer.fit(model)
    assert recorder.events == ["begin", "end"], recorder.events


@pytest.mark.integration
def test_validation_metrics_are_not_computed_twice():
    """epoch 级指标只能算一次：第二次会在已 reset 的累加器上得到全 0。

    v1.0.0 的 history.csv 里那列裸 ``accuracy`` 就是这么来的：fit() 在
    ``_evaluate`` 之外又调了一次 ``on_validation_epoch_end()``。
    现在指标以 ``val_`` 前缀并入，且只算一次。
    """
    trainer = _trainer(n=8, batch_size=4)
    history = trainer.fit(SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))["history"][-1]

    assert "val_accuracy" in history
    assert "accuracy" not in history, "epoch 级指标不应再以裸名出现（避免与 train_/val_ 混淆）"
    # 全 0 就是「二次计算」的症状
    assert history["val_accuracy"] > 0.0
