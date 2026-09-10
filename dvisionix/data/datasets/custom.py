# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 用户级自定义数据集（最简模板）。
"""用户级自定义数据集（最简模板）。

只要按 ``Sample`` 契约组织一个 samples 列表（每个 dict 必含 ``image``，按任务含
``label`` / ``boxes``+``labels`` / ``mask``），就可以直接传入，不需要写新类。
"""

from typing import Any, Callable, Dict, List, Optional

from ...registry import DATASETS
from ..base import BaseDataset
from ..collate import detection_collate, segmentation_collate


@DATASETS.register()
@DATASETS.register(name="custom")
class CustomDataset(BaseDataset):
    """最简自定义数据集：传入 samples 列表 + transforms 即可。

    Examples:
        >>> samples = [{"image": "a.jpg", "label": 0}, ...]
        >>> ds = CustomDataset(samples, transforms=my_pipeline)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transforms: Optional[Callable] = None,
        task_type: str = "classification",
        collate_fn: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        if not samples:
            raise ValueError("samples 不能为空。")
        super().__init__(samples, transforms=transforms, return_meta=return_meta)
        self.task_type = task_type
        if collate_fn is not None:
            self.collate_fn = collate_fn
        elif task_type == "detection":
            self.collate_fn = detection_collate
        elif task_type == "segmentation":
            self.collate_fn = segmentation_collate
        # 构造期自检：collate_fn 不可调用时，DataLoader 要等到取第一个 batch 才炸，
        # 定位成本高。注意这里**不要**用 staticmethod 包装 ——
        # 实例属性不经过描述符协议，staticmethod 对象只在 Python 3.10+ 才可调用，
        # 直接存函数对象才是正确且无版本依赖的写法。
        if self.collate_fn is not None and not callable(self.collate_fn):
            raise TypeError(
                f"collate_fn 必须是可调用对象，当前为 {type(self.collate_fn).__name__}"
                f"（task_type={task_type!r}）"
            )


__all__ = ["CustomDataset"]
