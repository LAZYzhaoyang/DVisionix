# -*- coding: utf-8 -*-
"""变换基类与组合管道。

设计：
- BaseTransform：所有变换的基类，定义 ``__call__(Sample) -> Sample`` 接口，
  通过 ``@TRANSFORMS.register()`` 注册到全局 ``TRANSFORMS``，可通过配置驱动构建。
- TransformPipeline：组合器，按顺序应用多个变换，自动聚合 ``provides_normalization``。

``provides_normalization``：
- 设为 True 表示该变换已完成 mean/std 归一化（如 ImageNormalize）。
- ``BaseDataset`` 在拼装 dataset 时会检查 pipeline 是否声明归一化，
  避免在 dataset 里再补一次导致二次归一化。
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from ..sample import Sample
from ...registry import TRANSFORMS


class BaseTransform:
    """所有变换的基类。

    约定：
    - 输入输出都是 ``Sample``（dict），至少含 ``image`` 字段；
    - 按字段缺失则不处理（如 bbox 变换对没有 ``boxes`` 字段的 sample 静默跳过）；
    - 子类实现 ``__call__`` 即可，无须继承 nn.Module。
    """

    name: str = ""
    provides_normalization: bool = False

    def __call__(self, sample: Sample) -> Sample:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TransformPipeline:
    """变换组合器——按顺序应用多个原子变换。

    Args:
        transforms: 原子变换列表（实例 / 配置字典 / 字符串，可混用）。
    """

    def __init__(self, transforms: Optional[Sequence[Any]] = None):
        from .builder import build_transform  # 避免循环依赖
        if transforms is None:
            self.transforms: List[BaseTransform] = []
        else:
            self.transforms = [build_transform(t) for t in transforms]
        self.provides_normalization = any(
            getattr(t, "provides_normalization", False) for t in self.transforms
        )

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms)

    def append(self, transform: Any) -> "TransformPipeline":
        from .builder import build_transform
        self.transforms.append(build_transform(transform))
        self.provides_normalization = (
            self.provides_normalization or getattr(self.transforms[-1], "provides_normalization", False)
        )
        return self

    def __repr__(self) -> str:
        body = ",\n    ".join(repr(t) for t in self.transforms)
        return f"TransformPipeline([\n    {body}\n])"


__all__ = ["BaseTransform", "TransformPipeline"]