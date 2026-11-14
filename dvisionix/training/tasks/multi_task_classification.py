# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 多任务分类任务（共享骨干 + 每任务一个头，按样本路由）。
"""多任务分类任务（MultiTaskClassificationTask，契约 P3-2）。

batch 契约：``image (B, C, H, W)``、``label (B,) long``、``task`` 为长度 B 的
``list[str]``（``DataLoader`` 默认 collate 产出）。每个样本按 ``task`` 路由到模型
对应分类头，各任务用**自己的子集**独立算交叉熵与指标。

两条硬约束：

1. **验证步 preds / targets 的键恒等于模型的全部任务名**：本 batch 未出现的任务
   用 ``(0, C_task)`` 空 logits 与 ``(0,)`` 空标签占位。各 rank 的 batch 任务组成
   常常不同，键集合一旦不一致，Trainer 的 ``all_gather`` 合并就会抛
   ``DDP gathered dictionaries have inconsistent keys``。
2. **指标按任务惰性创建**：任务第一次出现（且非空）时用 ``C = logits.shape[1]``
   构建 ``get_preset_metrics("classification", num_classes=C)``；epoch 结束时返回
   ``<task>_<metric>`` 前缀键（如 ``licence_accuracy``、``logo_f1``）并 reset。

多卡训练提示：每个 batch 未必覆盖全部头，DDP 需要 ``find_unused_parameters: true``；
单卡训练不受影响（契约 P3 的验收口径是单卡）。
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ...errors import ConfigError, TaskDataMismatchError
from ...metrics import BaseMetric, MetricCollection, get_preset_metrics
from ...models.losses import CrossEntropy, compute_loss
from .base import BaseTask, _merge_legacy_hyperparams

#: 指标配置里的"选择器"键：出现这些键说明该字典是单条指标配置而非 {任务: 模板}。
_METRIC_SELECTOR_KEYS = ("type", "_name_")


# ---------------------------------------------------------------------------
# 模块级工具
# ---------------------------------------------------------------------------
def _unwrap_model(model: Any) -> Any:
    """剥掉 DDP / ``torch.compile`` 的包装，拿到真实模型。

    验证步的键全集依赖 ``model.task_names``，而 Trainer 传给任务的是包装后的模型；
    DDP 的 ``task_names`` 不在包装层上，读不到就会退化成"按本 batch 任务取键"，
    各 rank 键不一致 → ``all_gather`` 崩溃。

    与 ``training.trainer._unwrap_module`` 同义；这里不能直接复用，因为 trainer
    在模块级 import 了 ``tasks``，反向 import 会形成循环依赖。
    """
    seen: set[int] = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        inner = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        if inner is None or inner is model:
            break
        model = inner
    return model


def _as_task_list(value: Any) -> list[str]:
    """把 ``batch["task"]`` 规整成 ``list[str]``。"""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TaskDataMismatchError(
        f"batch['task'] 必须是 str 或 list[str]（当前为 {type(value).__name__}）",
        hint="多任务分类数据集的每个样本带 task 字段，DataLoader 的默认 collate "
        "会把它拼成 list[str]；请检查 data.spec 的 data_type 与 collate_fn。",
    )


def _group_indices(task_list: Sequence[str]) -> dict[str, list[int]]:
    """按任务名分组样本下标（组内升序，与模型内部的特征子集顺序一致）。"""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(task_list):
        groups.setdefault(name, []).append(index)
    return groups


def _index_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
    """把下标列表转成设备上的 long 张量（``index_select`` 会保留顺序）。"""
    return torch.as_tensor(list(indices), dtype=torch.long, device=device)


def _normalize_task_loss_weights(weights: Any) -> dict[str, float]:
    """校验并规整 ``task_loss_weights``（未给出的任务按 1.0 计）。"""
    if weights is None:
        return {}
    if not isinstance(weights, Mapping):
        raise ConfigError(
            f"task_loss_weights 必须是 {{任务名: 权重}} 字典（当前为 {type(weights).__name__}）",
            hint='示例：task_loss_weights={"licence": 1.0, "logo": 0.5}。',
        )
    result: dict[str, float] = {}
    for name, value in weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"task_loss_weights[{name!r}] 必须是数值（当前为 {value!r}）",
                hint="权重用于缩放该任务的交叉熵，请填非负浮点数。",
            ) from exc
        if weight < 0:
            raise ConfigError(
                f"task_loss_weights[{name!r}] 不能为负数（当前为 {weight}）",
                hint="负权重会把该任务的损失变成奖励信号；如需忽略某任务请填 0.0。",
            )
        result[str(name)] = weight
    return result


def _split_metric_spec(metrics: Any) -> tuple[dict[str, Any], Any]:
    """拆分 ``metrics`` 参数：``({任务: 模板}, 共用模板)``。

    ``metrics`` 的三种写法：

    - ``None``：每个任务用分类预设组合；
    - ``{任务名: 指标模板}``：逐任务指定模板（键不能是 ``type`` / ``_name_``）；
    - 其它（``MetricCollection`` / 指标列表 / 单条指标配置）：所有任务共用同一份模板，
      每个任务各自构建独立实例。想用 ``{"accuracy": {...}}`` 这类"指标名 -> 参数"
      写法时请包成 list，否则会被当成逐任务映射。
    """
    if metrics is None:
        return {}, None
    if isinstance(metrics, Mapping) and metrics:
        keys = [str(key) for key in metrics]
        if all(key not in _METRIC_SELECTOR_KEYS for key in keys):
            return {str(key): value for key, value in metrics.items()}, None
    return {}, metrics


def _build_metric_collection(spec: Any, num_classes: int) -> MetricCollection:
    """按任务的类别数构建指标集合；``spec=None`` 时用分类预设组合。"""
    if spec is None:
        return get_preset_metrics("classification", num_classes=num_classes)
    if isinstance(spec, MetricCollection):
        # 已建好的集合带累加器状态，逐任务必须深拷贝，否则各任务指标互相污染。
        return copy.deepcopy(spec)
    if isinstance(spec, BaseMetric):
        return MetricCollection(metrics=[copy.deepcopy(spec)])
    if isinstance(spec, (str, Mapping)):
        return MetricCollection(metrics=[spec])
    return MetricCollection(metrics=list(spec))


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------
class MultiTaskClassificationTask(BaseTask):
    """多任务分类任务：路由式共享骨干 + 每任务一个分类头。

    Args:
        num_classes: ``{任务名: 类别数}``（可选；模型自带类别数时留空）。只在模型
            没有 ``num_classes`` 字典时用于给"本 batch 未出现任务"的空 logits 定宽。
        task_loss_weights: ``{任务名: 权重}``，缩放该任务的交叉熵后再求和；未列出的
            任务权重为 1.0（契约 P3-2.2）。
        learning_rate: 兼容旧 API，等价于写 ``optimizer_cfg["lr"]``。
        weight_decay: 兼容旧 API，等价于写 ``optimizer_cfg["weight_decay"]``。
        optimizer_cfg: 优化器配置，默认 ``{"type": "adam", "lr": 1e-3}``。
        scheduler_cfg: 调度器配置，默认 ReduceLROnPlateau(monitor="val_loss")。
        loss: 损失模块 / 配置；默认所有任务共享一个 ``CrossEntropy``。
        metrics: 指标模板。``None`` 时每个任务用 ``get_preset_metrics`` 的分类预设；
            ``{任务名: 模板}`` 可逐任务指定；其它形态作为所有任务的共用模板。
    """

    task_type = "multi_task_classification"
    accepted_model_tasks = ("multi_task_classification",)
    accepted_data_types = ("multi_task_classification",)
    required_batch_fields = ("image", "label", "task")

    @classmethod
    def build_transforms(cls, train: bool, image_size: int):
        """返回多任务分类的预设变换（复用 ``ClassificationTransforms``）。"""
        from ...data.transforms import ClassificationTransforms

        return ClassificationTransforms(train=train, image_size=image_size)

    def __init__(
        self,
        num_classes: dict | None = None,
        task_loss_weights: dict[str, float] | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        loss: Any = None,
        metrics: Any = None,
    ) -> None:
        # BaseTask 只支持单个 MetricCollection；多任务指标由本类按任务自建并管理。
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=None)
        self.num_classes = num_classes
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        self.task_loss_weights = _normalize_task_loss_weights(task_loss_weights)
        if self.loss is None:
            self.loss = CrossEntropy()

        #: ``{任务名: MetricCollection}``，任务第一次出现时惰性创建并常驻（跨 epoch 复用）。
        self.metrics: dict[str, MetricCollection] = {}
        self._metric_num_classes: dict[str, int] = {}
        self._metric_templates, self._shared_metric_spec = _split_metric_spec(metrics)

    # ------------------------------------------------------------------
    # 训练 / 验证步（契约 P3-2.2 / P3-2.3）
    # ------------------------------------------------------------------
    def training_step(
        self, model: nn.Module, batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """单步训练：路由式前向 + 逐任务交叉熵，返回 ``{loss, <task>_loss, <task>_acc}``。

        ``loss`` 是各任务交叉熵按 ``task_loss_weights`` 加权后的**和**；
        ``<task>_loss`` 记录该任务**未加权**的交叉熵，便于直接对比各任务难度。
        键随本 batch 实际出现的任务变化（训练日志本就允许键变化）。
        """
        images = self.to_device(batch["image"], device)
        labels = self.to_device(batch["label"], device)
        task_list = _as_task_list(batch["task"])
        logits_by_task = model(images, task=task_list)

        groups = _group_indices(task_list)
        total_loss: torch.Tensor | None = None
        extras: dict[str, Any] = {}
        for name, logits in logits_by_task.items():
            sub_labels = labels.index_select(0, _index_tensor(groups[name], labels.device))
            task_loss, _ = compute_loss(self.loss, logits, sub_labels)
            weight = self.task_loss_weights.get(name, 1.0)
            weighted = task_loss * weight
            total_loss = weighted if total_loss is None else total_loss + weighted
            with torch.no_grad():
                predictions = logits.detach().argmax(dim=1)
                extras[f"{name}_loss"] = task_loss.detach()
                extras[f"{name}_acc"] = (predictions == sub_labels).float().mean()

        if total_loss is None:
            # 本 batch 没有任何可识别任务：返回可反传的零损失而不是崩掉训练循环。
            # 这通常意味着 data.spec 的任务名与模型 heads 不匹配，日志里 loss 恒为 0
            # 即为信号（训练步只警告一次，避免逐 batch 刷屏）。
            warnings.warn(
                "MultiTaskClassificationTask 的训练 batch 没有任何与模型头匹配的任务"
                f"（batch 任务: {sorted(set(task_list))}），本步损失为 0；"
                "请检查 data.spec 的任务名与 model.heads 是否一致。",
                UserWarning,
                stacklevel=2,
            )
            total_loss = torch.zeros((), device=images.device, requires_grad=True)
        return {"loss": total_loss, **extras}

    def validation_step(
        self, model: nn.Module, batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """单步验证：返回 ``{loss, preds, targets}``，且两个字典的键恒为**全部任务名**。

        本 batch 未出现的任务用 ``(0, C_task)`` 空 logits 与 ``(0,)`` 空标签占位，
        保证各 rank 的 dict 键集合一致（DDP ``all_gather`` 的硬要求）。

        ``loss`` 与训练步采用**同一加权口径**（``task_loss_weights`` 生效），
        这样 ``val_loss`` 与 ``train_loss`` 可比，早停 / 调度器监视的是同一个目标。
        """
        images = self.to_device(batch["image"], device)
        labels = self.to_device(batch["label"], device)
        task_list = _as_task_list(batch["task"])
        with torch.no_grad():
            logits_by_task = model(images, task=task_list)

        preds, targets = self._full_outputs(model, task_list, labels, images, logits_by_task)
        total_loss: torch.Tensor | None = None
        for name, logits in logits_by_task.items():
            task_loss, _ = compute_loss(self.loss, logits, targets[name])
            weighted = task_loss * self.task_loss_weights.get(name, 1.0)
            total_loss = weighted if total_loss is None else total_loss + weighted
        if total_loss is None:
            total_loss = images.new_zeros(())
        return {"loss": total_loss, "preds": preds, "targets": targets}

    # ------------------------------------------------------------------
    # 指标（契约 P3-2.4）
    # ------------------------------------------------------------------
    def reset_metrics(self) -> None:
        """重置全部任务的指标集合（集合本身保留，跨 epoch 复用同一批指标实例）。"""
        metrics = getattr(self, "metrics", None)
        if isinstance(metrics, dict):
            for collection in metrics.values():
                collection.reset()

    def update_metrics(self, preds: Any, targets: Any) -> None:
        """用 ``{任务: logits}`` 与 ``{任务: labels}`` 更新各任务指标（空张量跳过）。"""
        if not isinstance(preds, Mapping) or not isinstance(targets, Mapping):
            raise TaskDataMismatchError(
                "多任务验证的 preds / targets 必须是 {任务名: 张量} 字典"
                f"（当前为 {type(preds).__name__} / {type(targets).__name__}）",
                hint="请确认任务是 MultiTaskClassificationTask，"
                "且模型的 forward 返回 {任务名: logits}。",
            )
        if set(preds) != set(targets):
            raise TaskDataMismatchError(
                "多任务验证的 preds / targets 任务名集合不一致："
                f"preds={sorted(preds)}，targets={sorted(targets)}",
                hint="验证步必须为每个任务同时给出 logits 与标签（空任务用空张量占位）。",
            )
        for task, logits in preds.items():
            if not isinstance(logits, torch.Tensor):
                raise TaskDataMismatchError(
                    f"任务 {task!r} 的 preds 必须是张量（当前为 {type(logits).__name__}）",
                    hint="验证步的 preds / targets 约定为 {任务名: 张量}；"
                    "若模型改动了输出结构，请同步实现自定义任务。",
                )
            if logits.shape[0] == 0:
                continue  # 本 batch 该任务没有样本：不创建也不更新，避免污染指标
            collection = self._ensure_metrics(task, int(logits.shape[1]))
            collection.update(logits, targets[task])

    def on_validation_epoch_end(self) -> dict[str, Any]:
        """计算各任务指标并 reset，返回 ``<task>_<metric>`` 前缀键。"""
        result: dict[str, Any] = {}
        for task, collection in self.metrics.items():
            for name, value in collection.compute().items():
                result[f"{task}_{name}"] = value
            collection.reset()
        return result

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _ensure_metrics(self, task: str, num_classes: int) -> MetricCollection:
        """惰性创建 / 取回某任务的指标集合（``C = logits.shape[1]``）。"""
        collection = self.metrics.get(task)
        if collection is not None:
            cached = self._metric_num_classes.get(task)
            if cached != num_classes:
                raise TaskDataMismatchError(
                    f"任务 {task!r} 的类别数在验证期间发生变化（{cached} -> {num_classes}），"
                    "同一 epoch 内指标无法复用同一套累加器",
                    hint="请检查该任务的标签空间与分类头的 num_classes 是否一致。",
                )
            return collection
        spec = self._metric_templates.get(task)
        if spec is None:
            spec = self._shared_metric_spec
        collection = _build_metric_collection(spec, num_classes)
        self.metrics[task] = collection
        self._metric_num_classes[task] = num_classes
        return collection

    def _full_outputs(
        self,
        model: nn.Module,
        task_list: Sequence[str],
        labels: torch.Tensor,
        images: torch.Tensor,
        logits_by_task: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """补齐"本 batch 未出现任务"的空张量，返回键全集一致的 preds / targets。"""
        groups = _group_indices(task_list)
        num_classes = self._num_classes_by_task(model, task_list)
        preds: dict[str, torch.Tensor] = {}
        targets: dict[str, torch.Tensor] = {}
        for name in self._task_universe(model, task_list):
            if name in logits_by_task:
                preds[name] = logits_by_task[name]
                targets[name] = labels.index_select(
                    0, _index_tensor(groups.get(name, ()), labels.device)
                )
                continue
            # 空占位：宽度取模型声明的类别数，保证各 rank 的 torch.cat 能对齐。
            preds[name] = images.new_zeros((0, int(num_classes.get(name, 0))))
            targets[name] = labels.new_zeros((0,), dtype=torch.long)
        return preds, targets

    def _task_universe(self, model: nn.Module, task_list: Sequence[str]) -> tuple[str, ...]:
        """任务名全集：优先模型的 ``task_names``，否则取本 batch 任务去重排序。"""
        names = getattr(_unwrap_model(model), "task_names", None)
        if names:
            return tuple(str(name) for name in names)
        return tuple(sorted(set(task_list)))

    def _num_classes_by_task(self, model: nn.Module, task_list: Sequence[str]) -> dict[str, int]:
        """模型（其次任务自身）声明的 ``{任务: 类别数}``；都没有时返回空字典。"""
        for source in (_unwrap_model(model), self):
            values = getattr(source, "num_classes", None)
            if isinstance(values, Mapping):
                return {str(name): int(count) for name, count in values.items()}
        return {}


__all__ = ["MultiTaskClassificationTask"]
