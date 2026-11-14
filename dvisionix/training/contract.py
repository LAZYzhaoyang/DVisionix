# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 任务 / 模型 / 数据三方契约校验（P2-3）。
"""任务 / 模型 / 数据三方契约校验（P2-3）。

训练开始前把"分割任务配分类模型"这类静默错配变成**可操作**的错误：

- :func:`check_task_model`：任务要求的模型能力与模型声明的能力必须有交集，
  且数据集与模型的 ``num_classes`` 必须一致；
- :func:`check_task_data`：DVX 数据集的 ``data_type`` 必须在任务接受集合内
  （没有 ``data_type`` 属性的自定义 / 公开数据集直接放行）；
- :func:`check_batch_fields`：训练 batch 必须携带任务声明的必需字段，
  在**第一个训练 batch** 上校验一次即可，避免每步开销。

所有错误都使用 :mod:`dvisionix.errors` 中的领域异常，消息遵循
"发生了什么 + 如何修复" 的统一格式。
"""

from typing import Any

from ..errors import TaskDataMismatchError, TaskModelMismatchError
from ..models.base import model_task_types


def _format_abilities(values: Any) -> str:
    """把能力集合格式化成稳定的可读文本（空集合显示为"未声明"）。"""
    items = tuple(values or ())
    return "、".join(repr(item) for item in items) if items else "未声明（空）"


def check_task_model(task: Any, model: Any, dataset: Any = None) -> None:
    """校验任务与模型（以及可选的数据集）是否匹配。

    规则（契约 P2-3.1）：

    1. 模型能力（:func:`dvisionix.models.base.model_task_types`）与
       ``task.accepted_model_tasks`` 的交集为空 → ``TaskModelMismatchError``；
    2. ``dataset`` 与 ``model`` 都暴露 ``num_classes`` 且不相等 →
       ``TaskDataMismatchError``。

    Args:
        task: 任务实例（``BaseTask`` 或其子类）。
        model: 模型实例（``nn.Module``）。
        dataset: 训练数据集；为 None 时跳过 ``num_classes`` 校验。

    Raises:
        TaskModelMismatchError: 任务与模型能力不匹配。
        TaskDataMismatchError: 数据集与模型的类别数不一致。
    """
    task_name = type(task).__name__
    model_name = type(model).__name__
    model_tasks = model_task_types(model)
    accepted = tuple(getattr(task, "accepted_model_tasks", ()) or ())

    # 自定义任务可能未声明能力：此时无从判断，跳过能力校验（渐进式契约）。
    # 内置任务均已显式声明，因此"分类模型配分割任务"仍会被拦下；
    # num_classes 校验与能力声明无关，继续执行。
    if accepted and not set(model_tasks) & set(accepted):
        raise TaskModelMismatchError(
            f"任务 {task_name} 与模型 {model_name} 不匹配："
            f"模型声明的任务能力为 [{_format_abilities(model_tasks)}]，"
            f"任务要求的能力为 [{_format_abilities(accepted)}]，两者没有交集。",
            hint=(
                f"请让模型支持 {list(accepted)} 之一（设置模型的 task_type 或 "
                f"supported_tasks），或改用与模型能力 {list(model_tasks)} 匹配的任务；"
                f"并检查配置中的 task.type 与 model.type 是否配对。"
            ),
        )

    if dataset is None:
        return
    dataset_num_classes = getattr(dataset, "num_classes", None)
    model_num_classes = getattr(model, "num_classes", None)
    if (
        dataset_num_classes is not None
        and model_num_classes is not None
        and int(dataset_num_classes) != int(model_num_classes)
    ):
        raise TaskDataMismatchError(
            f"数据集 {type(dataset).__name__} 的 num_classes={int(dataset_num_classes)} "
            f"与模型 {model_name} 的 num_classes={int(model_num_classes)} 不一致。",
            hint=(
                "请让配置中的 model.num_classes 等于数据集类别数"
                "（分类/分割的类别表通常来自 dvx_dataset.yaml 的 classes），"
                "或确认 data.spec 指向了正确的数据集。"
            ),
        )


def check_task_data(task: Any, dataset: Any) -> None:
    """校验任务与数据集 ``data_type`` 是否匹配。

    没有 ``data_type`` 属性的数据集（``CustomDataset`` / 公开数据集预设）不携带
    数据类型声明，直接放行；这是为了兼容 P2 之前的数据接入路径。

    Args:
        task: 任务实例（``BaseTask`` 或其子类）。
        dataset: 训练数据集。

    Raises:
        TaskDataMismatchError: 数据集的 ``data_type`` 不在任务接受集合内。
    """
    data_type = getattr(dataset, "data_type", None)
    if not data_type:
        return
    accepted = tuple(getattr(task, "accepted_data_types", ()) or ())
    if not accepted:
        # 自定义任务未声明 accepted_data_types：无从判断，放行（渐进式契约）。
        return
    if data_type not in accepted:
        raise TaskDataMismatchError(
            f"数据集 {type(dataset).__name__} 的 data_type={data_type!r} "
            f"与任务 {type(task).__name__} 不匹配："
            f"任务接受的数据类型为 [{_format_abilities(accepted)}]。",
            hint=(
                "请检查 data.spec 指向的 dvx_dataset.yaml 的 data_type，"
                f"或改用接受 {list(accepted)} 的任务；"
                "分类 / 检测 / 分割任务分别对应 ClassificationTask / DetectionTask / "
                "SegmentationTask。"
            ),
        )


def check_batch_fields(task: Any, batch: Any) -> None:
    """校验训练 batch 是否携带任务必需字段。

    只在**第一个训练 batch** 调用一次；失败说明数据管线（data_type / transforms /
    collate_fn）与任务不匹配，应在真正训练前停止。

    Args:
        task: 任务实例（``BaseTask`` 或其子类）。
        batch: DataLoader 产出的一个 batch。

    Raises:
        TaskDataMismatchError: batch 缺少任务声明的必需字段，或 batch 不是字典。
    """
    task_name = type(task).__name__
    required = tuple(getattr(task, "required_batch_fields", ("image",)) or ())
    if not isinstance(batch, dict):
        raise TaskDataMismatchError(
            f"任务 {task_name} 需要字典 batch，实际收到 {type(batch).__name__}。",
            hint="请检查 collate_fn 是否返回 dict（batch 应形如 {'image': ..., 'label': ...}）。",
        )
    missing = [field for field in required if field not in batch]
    if not missing:
        return
    raise TaskDataMismatchError(
        f"训练 batch 缺少任务 {task_name} 的必需字段 {missing}；"
        f"实际 batch 键为 {sorted(batch)}。",
        hint=(
            "请检查 data.dataset / data.spec 的 data_type 是否与任务一致，"
            "并确认 transforms 与 collate_fn 会产出这些字段"
            "（检测需 boxes/labels、分割需 mask、SimCLR 需 image1/image2）。"
        ),
    )


__all__ = ["check_task_model", "check_task_data", "check_batch_fields"]
