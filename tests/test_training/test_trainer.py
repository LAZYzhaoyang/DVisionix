# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 测试通用训练引擎和任务系统

# D:\\ZhaoyangProject\\DVisionix\\tests\\test_training\\test_trainer.py

"""
测试通用训练引擎和任务系统
"""

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.models import SimpleCNN
from dvisionix.training import (
    BaseTask,
    ClassificationTask,
    ModelCheckpoint,
    ProgressBar,
    Trainer,
)


class TestTaskSystem:
    """测试任务系统"""

    def test_classification_task(self):
        """测试分类任务"""
        task = ClassificationTask(num_classes=10, learning_rate=1e-3)

        # 测试 configure_optimizers
        model = SimpleCNN(num_classes=10)
        opt_config = task.configure_optimizers(model)
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

        # 测试 training_step
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "label": torch.randint(0, 10, (4,)),
        }
        device = torch.device("cpu")

        result = task.training_step(model, batch, device)
        assert "loss" in result
        assert "acc" in result
        assert isinstance(result["loss"], torch.Tensor)

    def test_custom_task(self):
        """测试自定义任务"""

        class MyCustomTask(BaseTask):
            def __init__(self):
                super().__init__()
                self.loss_fn = torch.nn.MSELoss()

            def training_step(self, model, batch, device):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = self.loss_fn(pred, y)
                return {"loss": loss}

            def validation_step(self, model, batch, device):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = self.loss_fn(pred, y)
                return {"loss": loss}

            def configure_optimizers(self, model):
                return torch.optim.Adam(model.parameters(), lr=1e-3)

        # 简单的回归模型
        class RegressionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        task = MyCustomTask()
        model = RegressionModel()

        # 测试优化器配置
        optimizer = task.configure_optimizers(model)
        assert isinstance(optimizer, torch.optim.Optimizer)

        # 测试训练步
        batch = {"x": torch.randn(4, 10), "y": torch.randn(4, 1)}
        device = torch.device("cpu")

        result = task.training_step(model, batch, device)
        assert "loss" in result


class TestCallbacks:
    """测试回调系统"""

    def test_progress_bar(self):
        """测试进度条回调"""
        callback = ProgressBar(log_interval=10)
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_batch_end")

    def test_model_checkpoint(self, tmp_path):
        """测试模型检查点回调"""
        callback = ModelCheckpoint(save_dir=str(tmp_path), monitor="val_loss", mode="min")
        assert hasattr(callback, "on_epoch_end")


class TestTrainer:
    """测试通用训练引擎"""

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        # 创建简单的数据集
        x = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))

        # 包装为字典格式的数据集
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return {"image": self.x[idx], "label": self.y[idx]}

        dataset = DictDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # 创建任务和训练器
        task = ClassificationTask(num_classes=10)
        trainer = Trainer(task, train_loader, max_epochs=2)

        assert trainer.max_epochs == 2
        assert trainer.task is task

    def test_trainer_fit(self):
        """测试训练流程"""
        # 小数据集快速测试
        x = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 10, (20,))

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return {"image": self.x[idx], "label": self.y[idx]}

        train_dataset = DictDataset(x, y)
        val_dataset = DictDataset(x[:10], y[:10])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)

        model = SimpleCNN(num_classes=10)
        task = ClassificationTask(num_classes=10)

        trainer = Trainer(
            task,
            train_loader,
            val_loader,
            max_epochs=1,
            log_interval=10,
        )

        result = trainer.fit(model)

        assert result is not None
        assert trainer.current_epoch == 0  # 0-based


class TestDDPAggregation:
    """DDP 结果聚合的递归语义（CodePlan 7.4 步骤 2-1 / 7.1.2 D5）。

    v1.0.0 的 `_concat_objects` 对 list 与 tuple 一律 `extend`，于是检测任务的
    `preds = (boxes_list, scores_list, labels_list)` 会被拍平成 6 元组，
    `task.update_metrics` 的三元解包随即可错 —— 而 286 条测试全绿也没发现。
    """

    def test_concat_tensors_along_batch_dim(self):
        from dvisionix.training.trainer import _concat_objects

        merged = _concat_objects([torch.zeros(2, 3), torch.ones(1, 3)])
        assert merged.shape == (3, 3)

    def test_concat_tuple_recurses_by_position(self):
        """检测任务的核心回归：3-tuple 必须仍是 3-tuple，而不是被拍平。"""
        from dvisionix.training.trainer import _concat_objects

        rank0 = ([torch.tensor([[1.0, 2.0, 3.0, 4.0]])], [torch.tensor([0])], [torch.tensor([0.9])])
        rank1 = ([torch.tensor([[5.0, 6.0, 7.0, 8.0]])], [torch.tensor([1])], [torch.tensor([0.8])])

        merged = _concat_objects([rank0, rank1])

        assert isinstance(merged, tuple) and len(merged) == 3
        boxes, labels, scores = merged
        assert len(boxes) == 2 and len(labels) == 2 and len(scores) == 2
        assert torch.equal(labels[1], torch.tensor([1]))
        assert torch.equal(scores[0], torch.tensor([0.9]))

    def test_concat_nested_dict_recurses_by_key(self):
        from dvisionix.training.trainer import _concat_objects

        merged = _concat_objects(
            [
                {"cls": torch.zeros(2, 3), "box": (torch.zeros(2, 4),)},
                {"cls": torch.ones(1, 3), "box": (torch.ones(1, 4),)},
            ]
        )
        assert merged["cls"].shape == (3, 3)
        assert isinstance(merged["box"], tuple)
        assert merged["box"][0].shape == (3, 4)

    def test_concat_namedtuple_preserves_type(self):
        import collections

        from dvisionix.training.trainer import _concat_objects

        Pair = collections.namedtuple("Pair", ["a", "b"])
        merged = _concat_objects([Pair(torch.zeros(1, 2), 1), Pair(torch.ones(1, 2), 1)])
        assert isinstance(merged, Pair)
        assert merged.a.shape == (2, 2) and merged.b == 1

    def test_concat_list_of_scalars_extends(self):
        from dvisionix.training.trainer import _concat_objects

        assert _concat_objects([[1, 2], [3]]) == [1, 2, 3]

    def test_concat_inconsistent_tuple_length_raises(self):
        from dvisionix.training.trainer import _concat_objects

        with pytest.raises(ValueError, match="tuples have inconsistent structure"):
            _concat_objects([(torch.zeros(1), torch.zeros(1)), (torch.zeros(1),)])

    def test_concat_inconsistent_dict_keys_raises(self):
        from dvisionix.training.trainer import _concat_objects

        with pytest.raises(ValueError, match="dictionaries have inconsistent keys"):
            _concat_objects([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}])

    def test_concat_inconsistent_scalars_raises(self):
        from dvisionix.training.trainer import _concat_objects

        with pytest.raises(ValueError, match="scalar objects have inconsistent values"):
            _concat_objects([1, 2])
        assert _concat_objects([1, 1]) == 1

    def test_concat_empty_returns_empty(self):
        from dvisionix.training.trainer import _concat_objects

        assert _concat_objects([]) == []

    def test_gather_returns_none_without_preds_or_targets(self):
        """没有 preds/targets 的任务（如 SimCLR）不应触发集合通信。"""
        from dvisionix.training.trainer import _gather_preds_targets

        assert _gather_preds_targets({"loss": 1.0}, world_size=2, rank=0) is None
        assert _gather_preds_targets({"preds": 1}, world_size=2, rank=0) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
