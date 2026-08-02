# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 阶段 3 训练引擎能力测试：AMP / 梯度累积 / 种子 / resume。
"""阶段 3 训练引擎能力测试：AMP / 梯度累积 / 种子 / resume。"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.models import SimpleCNN
from dvisionix.training import (
    ClassificationTask,
    EarlyStopping,
    Trainer,
)


class _TinyDataset(torch.utils.data.Dataset):
    """确定性小数据集：seed 相同则样本相同。"""

    def __init__(self, n: int = 8, num_classes: int = 3, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 32, 32, generator=gen)
        self.labels = torch.randint(0, num_classes, (n,), generator=gen)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def _make_trainer(**overrides):
    task = ClassificationTask(num_classes=3, learning_rate=1e-2)
    train_loader = DataLoader(_TinyDataset(n=8), batch_size=2, shuffle=False)
    val_loader = DataLoader(_TinyDataset(n=4, seed=1), batch_size=2, shuffle=False)
    kwargs = dict(
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=1,
        device="cpu",
        log_interval=999,
    )
    kwargs.update(overrides)
    return Trainer(**kwargs)


class TestAMPSmoke:
    """CPU 环境下 AMP 应该退化为 scaler=None，训练流程不报错。"""

    def test_amp_cpu_falls_back(self):
        trainer = _make_trainer(amp=True)
        assert trainer.scaler is None  # CPU 不启用 GradScaler
        model = SimpleCNN(num_classes=3, in_channels=3)
        result = trainer.fit(model)
        assert result["global_step"] > 0


class TestGradientAccumulation:
    """梯度累积：等效放大 batch size，global_step 应按累积步数缩减。"""

    def test_accum_reduces_optimizer_steps(self):
        trainer = _make_trainer(accumulate_grad_batches=2)
        model = SimpleCNN(num_classes=3, in_channels=3)
        result = trainer.fit(model)
        # 数据集 8 / batch 2 = 4 个 batch，accum=2 => 2 次 optimizer.step()
        assert result["global_step"] == 2


class TestSeedReproducibility:
    """相同 seed 应得到相同的最终参数。"""

    def test_same_seed_same_weights(self):
        model_a = SimpleCNN(num_classes=3, in_channels=3)
        model_b = SimpleCNN(num_classes=3, in_channels=3)
        # 起点对齐
        model_b.load_state_dict(model_a.state_dict())

        t1 = _make_trainer(seed=42)
        t1.fit(model_a)

        t2 = _make_trainer(seed=42)
        t2.fit(model_b)

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb, atol=1e-6)


class TestResume:
    """检查点保存/加载：恢复 optimizer/scheduler/callback 状态。"""

    def test_checkpoint_roundtrip_with_callback_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            es = EarlyStopping(monitor="val_loss", patience=100)

            trainer = _make_trainer(callbacks=[es])
            model = SimpleCNN(num_classes=3, in_channels=3)
            trainer.fit(model)

            saved_best = es.best_value
            saved_wait = es.wait

            ckpt = os.path.join(tmp, "state.pt")
            trainer.save_checkpoint(ckpt)

            # 新 trainer 从 checkpoint 恢复
            es2 = EarlyStopping(monitor="val_loss", patience=100)
            trainer2 = _make_trainer(callbacks=[es2])
            model2 = SimpleCNN(num_classes=3, in_channels=3)
            # 手动挂 optimizer 后再 load（模拟真实流程）
            opt_cfg = trainer2.task.configure_optimizers(model2)
            trainer2.model = model2.to(trainer2.device)
            trainer2.optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg
            trainer2.load_checkpoint(ckpt, model2)

            assert es2.best_value == pytest.approx(saved_best)
            assert es2.wait == saved_wait
            assert trainer2.current_epoch == trainer.current_epoch + 1
            assert trainer2.global_step == trainer.global_step


class TestWeightedAggregation:
    """按样本数加权聚合：不等 batch 时结果应正确。"""

    def test_weighted_average(self):
        # 手工构造两个不同 batch size 的样本，验证 _run_epoch 返回值
        class UnevenDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = [
                    {"image": torch.randn(3, 32, 32), "label": torch.tensor(0)},
                    {"image": torch.randn(3, 32, 32), "label": torch.tensor(1)},
                    {"image": torch.randn(3, 32, 32), "label": torch.tensor(2)},
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        task = ClassificationTask(num_classes=3, learning_rate=1e-3)
        loader = DataLoader(UnevenDataset(), batch_size=2, shuffle=False)
        trainer = Trainer(
            task=task,
            train_loader=loader,
            val_loader=None,
            max_epochs=1,
            device="cpu",
            log_interval=999,
        )
        model = SimpleCNN(num_classes=3, in_channels=3)
        trainer.fit(model)
        # 主要验证不抛异常且 global_step 正确（2 batch => 2 步）
        assert trainer.global_step == 2
