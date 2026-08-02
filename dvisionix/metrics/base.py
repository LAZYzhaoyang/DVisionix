# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 指标基类定义。
"""指标基类定义。

所有指标（原子指标或组合指标）都继承 ``BaseMetric``，遵循累积式接口：

- ``reset()``：清空内部累积状态（每个 epoch 开始时调用）。
- ``update(*args, **kwargs)``：喂入一个 batch，仅累加内部状态，不返回最终值。
- ``compute()``：基于当前累积的全部状态计算并返回结果。

约定：
- 原子指标的 ``compute()`` 返回标量 ``float``（per-class 模式可返回 ``list``）。
- 组合/预设指标的 ``compute()`` 返回 ``dict``（key 为各指标名）。
- ``MetricCollection`` 汇总时：成员返回 dict 则合并其键，返回标量则以成员 `
ame`` 为键。

为什么用累积式而非"逐 batch 求平均"：mIoU / mAP / macro-F1 等指标
``mean(每个 batch 的值) != 全局值``，必须先累积混淆矩阵 / TP-FP 再计算。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

MetricValue = Union[float, list, Dict[str, Any]]


class BaseMetric(ABC):
    """所有指标的基类。

    Args:
        name: 指标名称，作为 ``MetricCollection`` 汇总结果时的 key。
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """喂入一个 batch，累加内部状态。子类必须实现。"""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> MetricValue:
        """基于当前累积状态计算并返回指标结果。子类必须实现。"""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """重置所有累积状态。子类必须实现。"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> MetricValue:
        """更新并返回当前指标值（便于单 batch 快速计算）。"""
        self.update(*args, **kwargs)
        return self.compute()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
