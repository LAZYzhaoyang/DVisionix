# -*- coding: utf-8 -*-
"""指标组合容器。

MetricCollection 把多个原子指标组合在一起，把 reset/update 广播给每个成员，
compute 汇总为一个 dict。支持三种成员写法（可混用）：

- BaseMetric 实例：直接使用。
- 配置字典：如 {"type": "f1_score", "average": "macro", "num_classes": 10}，走 METRICS 构建。
- 字符串：如 "accuracy"，等价于 {"type": "accuracy"}（无参构建）。

同时保留向后兼容的旧签名 MetricCollection(task_type="classification", num_classes=10)，
内部转为对应任务的预设组合。
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from .base import BaseMetric
from ..registry import METRICS


MetricSpec = Union[BaseMetric, Dict[str, Any], str]


def build_single_metric(spec: MetricSpec) -> BaseMetric:
    """从实例 / 配置字典 / 字符串构建单个指标。"""
    if isinstance(spec, BaseMetric):
        return spec
    if isinstance(spec, str):
        return METRICS.build({"type": spec})
    if isinstance(spec, dict):
        return METRICS.build(dict(spec))
    raise TypeError(f"Unsupported metric spec: {type(spec)}")


class MetricCollection:
    """指标组合容器（把 reset/update/compute 广播给成员并汇总）。"""

    def __init__(
        self,
        metrics: Optional[Sequence[MetricSpec]] = None,
        *,
        task_type: Optional[str] = None,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            metrics: 指标成员列表（实例 / 配置字典 / 字符串，可混用）。
            task_type: 向后兼容参数，给定时取该任务预设组合（忽略 metrics）。
            num_classes: 预设组合所需类别数。
            **kwargs: 传给预设组合的其它参数。
        """
        if task_type is not None:
            from .presets import get_preset_metrics
            preset = get_preset_metrics(task_type, num_classes=num_classes, **kwargs)
            self.metrics: List[BaseMetric] = list(preset.metrics)
        else:
            if metrics is None:
                raise ValueError("MetricCollection 需要 metrics 列表或 task_type。")
            self.metrics = [build_single_metric(s) for s in metrics]
        self.reset()

    def add(self, spec: MetricSpec) -> "MetricCollection":
        """追加一个指标成员，返回自身以支持链式调用。"""
        self.metrics.append(build_single_metric(spec))
        return self

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """把同一批输入广播给每个成员的 update。"""
        for m in self.metrics:
            m.update(*args, **kwargs)

    def compute(self) -> Dict[str, Any]:
        """汇总所有成员结果为一个 dict。

        成员返回 dict 则合并其键；返回标量/列表则以成员 name 为键。
        """
        result: Dict[str, Any] = {}
        for m in self.metrics:
            value = m.compute()
            if isinstance(value, dict):
                result.update(value)
            else:
                result[m.name] = value
        return result

    def __len__(self) -> int:
        return len(self.metrics)

    def __iter__(self):
        return iter(self.metrics)

    def __repr__(self) -> str:
        names = [m.name for m in self.metrics]
        return f"MetricCollection({names})"


__all__ = ["MetricCollection", "build_single_metric"]