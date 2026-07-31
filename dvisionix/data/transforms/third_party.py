# -*- coding: utf-8 -*-
"""第三方数据增强库封装。

支持把 albumentations / kornia / torchvision.transforms.v2 等第三方库的变换
适配为 ``BaseTransform`` 协议（``__call__(Sample) -> Sample``），可以与内置
原子变换自由组合、配置驱动构建。

约定：
- 所有封装类继承 ``BaseTransform``，并通过 ``register_third_party(name, lib)``
  注册到全局 ``THIRD_PARTY_TRANSFORMS``（详见 ``builder.py``）。
- 默认在 ``__init__.py`` 里导入并注册 ``AlbumentationsWrapper``。
"""

from typing import Any, Dict

import numpy as np

from ..sample import Sample
from .base import BaseTransform
from ...registry import TRANSFORMS


@TRANSFORMS.register()
@TRANSFORMS.register(name="albumentations")
class AlbumentationsWrapper(BaseTransform):
    """把 ``albumentations.Compose`` 适配为 ``BaseTransform`` 协议。

    输入 sample 字段约定：
    - ``image``：numpy uint8 (H, W, C) RGB
    - ``boxes``：(N, 4) [x1, y1, x2, y2] float，**当 is_detection=True 时才会被处理**
    - ``labels``：(N,) int，与 boxes 一一对应
    - ``mask``：(H, W) numpy int，**当 is_segmentation=True 时才会被处理**

    其它字段原样透传。
    """

    name = "albumentations"

    def __init__(
        self,
        albu_transforms: Any,
        is_detection: bool = False,
        is_segmentation: bool = False,
        bbox_format: str = "pascal_voc",
    ):
        try:
            import albumentations as A
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "albumentations is not installed. Please install it with: pip install albumentations"
            ) from exc
        self._A = A
        self.is_detection = is_detection
        self.is_segmentation = is_segmentation
        self._bbox_format = bbox_format

        if is_detection and getattr(albu_transforms, "processors", None) is not None                 and "bboxes" not in albu_transforms.processors:
            transforms = list(albu_transforms.transforms)
            albu_transforms = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format=bbox_format, label_fields=["labels"]),
            )
        self.albu_transforms = albu_transforms

    def __call__(self, sample: Sample) -> Sample:
        kwargs: Dict[str, Any] = {"image": sample["image"]}
        if self.is_detection and "boxes" in sample and len(sample["boxes"]) > 0:
            kwargs["bboxes"] = sample["boxes"].tolist()
            kwargs["labels"] = sample.get("labels", np.zeros(len(sample["boxes"]), dtype=np.int64)).tolist()
        if self.is_segmentation and "mask" in sample:
            kwargs["mask"] = sample["mask"]

        out = self.albu_transforms(**kwargs)

        sample["image"] = out["image"]
        if self.is_detection and "bboxes" in out:
            sample["boxes"] = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            sample["labels"] = np.asarray(out.get("labels", []), dtype=np.int64)
        if self.is_segmentation and "mask" in out:
            sample["mask"] = np.asarray(out["mask"], dtype=np.int64)
        return sample