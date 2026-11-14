# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: MultiTaskClassificationTask 测试（训练步 / 验证步键全集 / 惰性指标，契约 P3-2）。
"""MultiTaskClassificationTask 测试（契约 P3-2）。

覆盖：

- 训练步：``loss`` 为各任务交叉熵的加权和，``<task>_loss`` / ``<task>_acc`` 只含
  本 batch 实际出现的任务；``task_loss_weights`` 生效；
- 验证步：``preds`` / ``targets`` 的键**恒等于模型全部任务名**，未出现任务为
  ``(0, C_task)`` / ``(0,)`` 空张量；DDP 包装层（``.module``）也能读到任务名全集；
- 指标：按任务惰性创建（``C = logits.shape[1]``）、空张量跳过、epoch 结束返回
  ``<task>_<metric>`` 前缀键并 reset、跨 epoch 复用同一批指标实例；
- 健壮性：空任务 batch（没有任何可识别任务）不报错；坏 batch / 键不一致报领域错误；
- Trainer 接线：单卡 1 epoch 的 history 里同时出现 ``licence_*`` 与 ``logo_*`` 指标。
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from torch.utils.data import DataLoader

from dvisionix.errors import TaskDataMismatchError
from dvisionix.models import MultiTaskClassifier
from dvisionix.training import Trainer, build_task
from dvisionix.training.tasks import MultiTaskClassificationTask

pytestmark = pytest.mark.unit

IMAGE_SIZE = 8
TASK_NAMES = ("licence", "logo")
NUM_CLASSES = {"licence": 3, "logo": 2}
STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 8, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 8, "out_channels": 16, "stride": 2},
]
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
DEVICE = torch.device("cpu")


class _MultiTaskDS(torch.utils.data.Dataset):
    """最小多任务数据集：按顺序轮换任务，标签落在该任务的类别空间内。"""

    def __init__(self, tasks=TASK_NAMES, n=6):
        self.tasks = list(tasks)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        name = self.tasks[index % len(self.tasks)]
        return {
            "image": torch.randn(3, IMAGE_SIZE, IMAGE_SIZE),
            "label": torch.tensor(index % NUM_CLASSES[name], dtype=torch.long),
            "task": name,
        }


def _model(num_classes=None):
    counts = dict(NUM_CLASSES) if num_classes is None else dict(num_classes)
    return MultiTaskClassifier(
        backbone={
            "type": "sequential_backbone",
            "stages": STAGES,
            "input_size": IMAGE_SIZE,
        },
        heads={name: {"num_classes": count} for name, count in counts.items()},
    )


def _task(**kwargs):
    return MultiTaskClassificationTask(**kwargs)


def _batch(tasks=("licence", "logo", "licence"), labels=None):
    """构造一个 batch；标签固定，便于断言 loss 的加权关系。"""
    task_list = list(tasks)
    if labels is None:
        labels = [0, 1, 2][: len(task_list)]
    return {
        "image": torch.randn(len(task_list), 3, IMAGE_SIZE, IMAGE_SIZE),
        "label": torch.tensor(labels, dtype=torch.long),
        "task": task_list,
    }


class _NoTaskModel(nn.Module):
    """没有任何可识别任务的模型：``forward`` 返回空字典。"""

    def forward(self, x, task=None, **kwargs):
        return {}


class _Wrapper(nn.Module):
    """模拟 DDP / torch.compile 的包装层（任务名要通过 ``.module`` 才能读到）。"""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, x, **kwargs):
        return self.module(x, **kwargs)


# ---------------------------------------------------------------------------
# 契约类属性
# ---------------------------------------------------------------------------


class TestTaskContract:
    """任务的契约声明与构建入口。"""

    def test_contract_attributes(self):
        assert MultiTaskClassificationTask.task_type == "multi_task_classification"
        assert MultiTaskClassificationTask.accepted_model_tasks == ("multi_task_classification",)
        assert MultiTaskClassificationTask.accepted_data_types == ("multi_task_classification",)
        assert MultiTaskClassificationTask.required_batch_fields == ("image", "label", "task")

    def test_build_transforms(self):
        for train in (True, False):
            transforms = MultiTaskClassificationTask.build_transforms(train, IMAGE_SIZE)
            assert type(transforms).__name__ == "ClassificationTransforms"
            assert transforms.train is train

    def test_build_task_from_registry(self):
        task = build_task({"type": "MultiTaskClassificationTask"})
        assert isinstance(task, MultiTaskClassificationTask)
        assert task.metrics == {}
        assert type(task.loss).__name__ == "CrossEntropy"


# ---------------------------------------------------------------------------
# 训练步
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """训练步：加权总损失 + 逐任务键。"""

    def test_returns_prefixed_keys_for_present_tasks(self):
        task = _task()
        model = _model().train()
        out = task.training_step(model, _batch(), DEVICE)

        assert set(out) == {"loss", "licence_loss", "licence_acc", "logo_loss", "logo_acc"}
        assert out["loss"].requires_grad
        out["loss"].backward()
        for name in TASK_NAMES:
            assert all(param.grad is not None for param in model.heads[name].parameters())

    def test_loss_is_sum_of_task_losses(self):
        task = _task()
        model = _model().train()
        out = task.training_step(model, _batch(), DEVICE)

        expected = float(out["licence_loss"]) + float(out["logo_loss"])
        assert float(out["loss"].detach()) == pytest.approx(expected, rel=1e-5)

    def test_task_loss_weights_scale_the_total(self):
        batch = _batch()
        model = _model().train()
        base = _task().training_step(model, batch, DEVICE)
        weighted = _task(task_loss_weights={"licence": 2.0, "logo": 0.0}).training_step(
            model, batch, DEVICE
        )

        assert float(weighted["loss"].detach()) == pytest.approx(
            2.0 * float(weighted["licence_loss"])
        )
        assert float(weighted["logo_loss"]) == pytest.approx(float(base["logo_loss"]))
        assert float(weighted["licence_loss"]) == pytest.approx(float(base["licence_loss"]))

    def test_keys_follow_tasks_in_batch(self):
        task = _task()
        model = _model().train()
        out = task.training_step(model, _batch(tasks=("licence", "licence")), DEVICE)

        assert set(out) == {"loss", "licence_loss", "licence_acc"}

    def test_absent_head_gets_no_gradient(self):
        """本 batch 没有的任务：不参与损失，也不产生梯度（多卡需 find_unused_parameters）。"""
        task = _task()
        model = _model().train()
        out = task.training_step(model, _batch(tasks=("licence", "licence")), DEVICE)
        out["loss"].backward()

        assert all(param.grad is not None for param in model.heads["licence"].parameters())
        assert all(param.grad is None for param in model.heads["logo"].parameters())


# ---------------------------------------------------------------------------
# 验证步：键全集
# ---------------------------------------------------------------------------


class TestValidationStep:
    """验证步：preds / targets 的键恒等于模型的全部任务名。"""

    def test_keys_are_full_task_set(self):
        task = _task()
        model = _model().eval()
        out = task.validation_step(model, _batch(tasks=("licence", "licence")), DEVICE)

        assert set(out) == {"loss", "preds", "targets"}
        assert tuple(out["preds"]) == TASK_NAMES
        assert tuple(out["targets"]) == TASK_NAMES
        # 出现的任务：真实 logits / 标签
        assert out["preds"]["licence"].shape == (2, NUM_CLASSES["licence"])
        assert out["targets"]["licence"].shape == (2,)
        # 未出现的任务：按该头宽度补空张量（DDP all_gather 要求键集合一致）
        assert out["preds"]["logo"].shape == (0, NUM_CLASSES["logo"])
        assert out["targets"]["logo"].shape == (0,)
        assert out["targets"]["logo"].dtype == torch.long
        assert out["loss"].ndim == 0

    def test_empty_placeholder_is_empty_for_all_tasks(self):
        task = _task()
        model = _model().eval()
        out = task.validation_step(model, _batch(tasks=("logo",), labels=[1]), DEVICE)

        assert out["preds"]["licence"].shape == (0, NUM_CLASSES["licence"])
        assert out["targets"]["licence"].shape == (0,)
        assert out["preds"]["logo"].shape == (1, NUM_CLASSES["logo"])
        assert out["targets"]["logo"].tolist() == [1]

    def test_ddp_wrapper_exposes_task_names(self):
        task = _task()
        model = _Wrapper(_model().eval())
        out = task.validation_step(model, _batch(tasks=("licence", "licence")), DEVICE)
        assert tuple(out["preds"]) == TASK_NAMES

    def test_model_with_single_head_uses_its_own_universe(self):
        task = _task()
        model = _model(num_classes={"licence": 3}).eval()
        out = task.validation_step(model, _batch(tasks=("licence",)), DEVICE)
        assert tuple(out["preds"]) == ("licence",)

    def test_validation_loss_uses_task_loss_weights(self):
        """验证 loss 与训练 loss 同一加权口径：权重全 0 时总损失恰为 0。"""
        task = _task(task_loss_weights={"licence": 0.0, "logo": 0.0})
        out = task.validation_step(_model().eval(), _batch(), DEVICE)

        assert float(out["loss"]) == 0.0

    def test_bad_batch_task_type_raises(self):
        task = _task()
        batch = _batch()
        batch["task"] = torch.tensor([0, 1, 0])
        with pytest.raises(TaskDataMismatchError, match="batch\\['task'\\]"):
            task.validation_step(_model(), batch, DEVICE)

    def test_step_outputs_merge_across_ranks_like_ddp(self):
        """模拟 DDP：两 rank 的 batch 任务组成不同，preds/targets 仍能按键合并。

        ``Trainer._concat_objects`` 对 dict 的合并要求各 rank 键集合完全一致，
        否则直接抛 "DDP gathered dictionaries have inconsistent keys"。
        """
        from dvisionix.training.trainer import _concat_objects

        task = _task()
        model = _model().eval()
        rank0 = task.validation_step(model, _batch(tasks=("licence", "licence")), DEVICE)
        rank1 = task.validation_step(model, _batch(tasks=("logo",)), DEVICE)

        merged_preds = _concat_objects([rank0["preds"], rank1["preds"]])
        merged_targets = _concat_objects([rank0["targets"], rank1["targets"]])

        assert tuple(merged_preds) == TASK_NAMES
        assert tuple(merged_targets) == TASK_NAMES
        # 空占位 (0, C) 与另一 rank 的真实张量 (n_i, C) 能沿 batch 维拼接
        assert merged_preds["licence"].shape == (2, NUM_CLASSES["licence"])
        assert merged_preds["logo"].shape == (1, NUM_CLASSES["logo"])
        assert merged_targets["licence"].shape == (2,)
        assert merged_targets["logo"].shape == (1,)

        task.update_metrics(merged_preds, merged_targets)
        assert set(task.on_validation_epoch_end()) == {
            f"{task_name}_{metric}" for task_name in TASK_NAMES for metric in METRIC_NAMES
        }


# ---------------------------------------------------------------------------
# 指标：惰性创建 / 前缀键 / 重置
# ---------------------------------------------------------------------------


class TestMetrics:
    """按任务惰性创建指标，epoch 结束返回前缀键。"""

    def test_metrics_created_lazily_with_head_width(self):
        task = _task()
        model = _model().eval()
        assert task.metrics == {}

        out = task.validation_step(model, _batch(tasks=("licence",)), DEVICE)
        task.update_metrics(out["preds"], out["targets"])

        assert set(task.metrics) == {"licence"}
        num_classes = {getattr(metric, "num_classes", None) for metric in task.metrics["licence"]}
        assert NUM_CLASSES["licence"] in num_classes

        out = task.validation_step(model, _batch(tasks=("logo",)), DEVICE)
        task.update_metrics(out["preds"], out["targets"])
        assert set(task.metrics) == {"licence", "logo"}
        logo_classes = {getattr(metric, "num_classes", None) for metric in task.metrics["logo"]}
        assert NUM_CLASSES["logo"] in logo_classes

    def test_empty_tensors_are_skipped(self):
        task = _task()
        model = _model().eval()
        out = task.validation_step(model, _batch(tasks=("licence",)), DEVICE)
        assert out["preds"]["logo"].shape[0] == 0

        task.update_metrics(out["preds"], out["targets"])

        assert "logo" not in task.metrics, "空张量不得创建（也不得更新）指标集合"
        assert task.on_validation_epoch_end().keys() == {f"licence_{name}" for name in METRIC_NAMES}

    def test_epoch_end_returns_task_prefixed_keys(self):
        task = _task()
        model = _model().eval()
        for batch in (_batch(tasks=("licence",)), _batch(tasks=("logo",))):
            out = task.validation_step(model, batch, DEVICE)
            task.update_metrics(out["preds"], out["targets"])

        metrics = task.on_validation_epoch_end()

        assert set(metrics) == {
            f"{task_name}_{metric}" for task_name in TASK_NAMES for metric in METRIC_NAMES
        }
        assert all(0.0 <= float(value) <= 1.0 for value in metrics.values())

    def test_epoch_end_resets_accumulators(self):
        task = _task()
        model = _model().eval()
        for batch in (_batch(tasks=("licence",)), _batch(tasks=("logo",))):
            out = task.validation_step(model, batch, DEVICE)
            task.update_metrics(out["preds"], out["targets"])

        first = task.on_validation_epoch_end()
        second = task.on_validation_epoch_end()

        assert set(first) == set(second)
        # reset 之后没有新样本：准确率回到 0（而不是沿用上一轮的累加器）
        assert second["licence_accuracy"] == 0.0
        assert second["logo_accuracy"] == 0.0

    def test_metric_instances_are_reused_across_epochs(self):
        task = _task()
        model = _model().eval()
        out = task.validation_step(model, _batch(), DEVICE)
        task.update_metrics(out["preds"], out["targets"])
        collection = task.metrics["licence"]
        task.on_validation_epoch_end()

        out = task.validation_step(model, _batch(), DEVICE)
        task.update_metrics(out["preds"], out["targets"])

        assert task.metrics["licence"] is collection, "同一任务跨 epoch 应复用指标实例"

    def test_reset_metrics_clears_state(self):
        task = _task()
        model = _model().eval()
        out = task.validation_step(model, _batch(), DEVICE)
        task.update_metrics(out["preds"], out["targets"])

        task.reset_metrics()

        assert set(task.on_validation_epoch_end()) == {
            f"{task_name}_{metric}" for task_name in TASK_NAMES for metric in METRIC_NAMES
        }
        assert task.on_validation_epoch_end()["licence_accuracy"] == 0.0

    def test_preds_targets_key_mismatch_raises(self):
        task = _task()
        with pytest.raises(TaskDataMismatchError, match="任务名集合不一致"):
            task.update_metrics(
                {"licence": torch.randn(2, NUM_CLASSES["licence"])},
                {"logo": torch.tensor([0, 1])},
            )

    def test_non_tensor_preds_raises(self):
        task = _task()
        with pytest.raises(TaskDataMismatchError, match="preds 必须是张量"):
            task.update_metrics({"licence": [0.1, 0.2]}, {"licence": torch.tensor([0])})

    def test_metric_spec_per_task_mapping(self):
        """``metrics`` 可按任务给定模板（这里用平均方式不是默认值来证明生效）。"""
        task = _task(metrics={"licence": [{"type": "accuracy"}]})
        model = _model().eval()
        for batch in (_batch(tasks=("licence",)), _batch(tasks=("logo",))):
            out = task.validation_step(model, batch, DEVICE)
            task.update_metrics(out["preds"], out["targets"])

        metrics = task.on_validation_epoch_end()

        assert set(metrics) == {
            "licence_accuracy",
            "logo_accuracy",
            "logo_f1",
            "logo_precision",
            "logo_recall",
        }


# ---------------------------------------------------------------------------
# 空任务 batch 的健壮性
# ---------------------------------------------------------------------------


class TestEmptyTaskBatch:
    """没有任何可识别任务时不得崩训练循环（DDP 键全集之外的退化场景）。"""

    def test_validation_step_does_not_raise(self):
        task = _task()
        out = task.validation_step(_NoTaskModel(), _batch(tasks=("licence", "logo")), DEVICE)

        assert float(out["loss"]) == 0.0
        assert set(out["preds"]) == set(TASK_NAMES)
        assert all(value.shape[0] == 0 for value in out["preds"].values())
        task.update_metrics(out["preds"], out["targets"])
        assert task.on_validation_epoch_end() == {}

    def test_training_step_returns_zero_loss(self):
        task = _task()
        with pytest.warns(UserWarning, match="没有任何与模型头匹配的任务"):
            out = task.training_step(_NoTaskModel(), _batch(tasks=("licence", "logo")), DEVICE)
        assert float(out["loss"].detach()) == 0.0
        out["loss"].backward()  # 可反传：训练循环不会在这里崩


# ---------------------------------------------------------------------------
# Trainer 接线
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    """单卡 1 epoch：history 同时出现两个任务的指标。"""

    def test_history_contains_task_prefixed_metrics(self):
        task = _task()
        model = _model()
        train_loader = DataLoader(_MultiTaskDS(n=6), batch_size=2)
        val_loader = DataLoader(_MultiTaskDS(n=6), batch_size=2)
        trainer = Trainer(
            task,
            train_loader,
            val_loader,
            max_epochs=1,
            device="cpu",
            log_interval=999,
        )

        result = trainer.fit(model)

        assert result["global_step"] > 0
        history = result["history"][0]
        for task_name in TASK_NAMES:
            assert f"train_{task_name}_loss" in history
            assert f"train_{task_name}_acc" in history
            assert f"val_{task_name}_accuracy" in history
            assert f"val_{task_name}_f1" in history
        assert "val_loss" in history
