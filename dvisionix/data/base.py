# -*- coding: utf-8 -*-
"""数据集基类与 Sample 契约。

设计：
- ``BaseDataset`` 继承 ``torch.utils.data.Dataset``，但只关心：samples + transforms。
- 内部不再做归一化 / ToTensor / 图像解码——全部交给 ``TransformPipeline``。
- 子类只需：
    1. 在 ``__init__`` 中构造 ``self.samples``（list[dict]），每个 sample 至少含 ``image`` 字段
       （可以是文件路径 str / numpy ndarray / torch.Tensor）。
    2. 如需自定义读图（如 mask 文件、特殊格式），覆盖 ``load_image(sample)``。
- 数据集按 task 在 sample 中放额外字段：分类放 ``label``，检测放 ``boxes``+``labels``，
  分割放 ``mask``，自定义任务自由扩展。
- ``BaseDataset`` 通过 ``@DATASETS.register()`` 注册到全局 ``DATASETS`` 注册表；
  也可通过 ``task_type`` / `
ame`` 等装饰器参数指定多个注册名。
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import DATASETS
from .sample import Sample
from .transforms import TransformPipeline

# collate_fn 可选：默认 None，DataLoader 会用 PyTorch 默认 collate；
# 检测/分割如需变长 pad，dataset 自己在 ``collate_fn`` 属性上挂。
CollateFn = Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]]


@DATASETS.register()
@DATASETS.register(name="base_dataset")
class BaseDataset(Dataset):
    """所有数据集的基类。

    Args:
        samples: 样本列表，每个元素是 dict，必含 ``image`` 字段（路径 / numpy / tensor），
            按任务含 ``label`` / ``boxes``+``labels`` / ``mask``，可扩展其它字段。
        transforms: ``TransformPipeline``（或可调用对象）。为 None 时只做 ``load_image``。
        load_image: 自定义图像加载函数 ``sample -> np.ndarray``；默认用 ``cv2.imread`` 读路径。
        collate_fn: 变长 pad 时挂上的 collate 函数（如 ``detection_collate``）。
        return_meta: 是否透传 ``meta`` 字段。

    子类标准写法（最简形态）::

        @DATASETS.register()
        class MyDataset(BaseDataset):
            def __init__(self, root, ...):
                samples = [{"image": "a.jpg", "label": 0}, ...]
                super().__init__(samples, transforms=my_pipeline)

    子类可额外设置 ``self.task_type = "classification"``，供上层按任务取预设 pipeline。
    """

    task_type: str = ""

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        transforms: Optional[Union[TransformPipeline, Callable[[Sample], Sample]]] = None,
        load_image: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
        collate_fn: CollateFn = None,
        return_meta: bool = False,
    ):
        self.samples: List[Dict[str, Any]] = list(samples)
        self.transforms = transforms
        self.load_image_fn = load_image
        self.collate_fn = collate_fn
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, sample: Dict[str, Any]) -> np.ndarray:
        """读取 ``image`` 字段为 numpy uint8 (H, W, C) RGB。"""
        if self.load_image_fn is not None:
            return self.load_image_fn(sample)
        img = sample.get("image")
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            return arr
        if isinstance(img, str):
            import cv2

            bgr = cv2.imread(img, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image: {img}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise TypeError(f"Unsupported image type: {type(img)}")

    def load_mask(self, sample: Dict[str, Any]) -> np.ndarray:
        """读取 ``mask`` 字段为 numpy int64 (H, W)（支持路径 / ndarray / Tensor）。"""
        mask = sample.get("mask")
        if isinstance(mask, np.ndarray):
            return mask
        if isinstance(mask, torch.Tensor):
            return mask.detach().cpu().numpy()
        if isinstance(mask, str):
            import cv2

            arr = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise FileNotFoundError(f"Failed to read mask: {mask}")
            return arr
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = dict(self.samples[index])
        if "image" in raw and not isinstance(raw["image"], (np.ndarray, torch.Tensor)):
            raw["image"] = self.load_image(raw)
        if "mask" in raw and isinstance(raw["mask"], str):
            raw["mask"] = self.load_mask(raw)
        if not self.return_meta and "meta" in raw:
            raw.pop("meta", None)
        if self.transforms is not None:
            raw = self.transforms(raw)
        return raw


__all__ = ["BaseDataset"]
