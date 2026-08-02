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
            self.collate_fn = staticmethod(detection_collate)
        elif task_type == "segmentation":
            self.collate_fn = staticmethod(segmentation_collate)


__all__ = ["CustomDataset"]
