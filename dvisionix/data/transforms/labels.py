# -*- coding: utf-8 -*-
"""标签字段的张量化与基础转换。"""

import torch

from ...registry import TRANSFORMS
from ..sample import Sample
from .base import BaseTransform


@TRANSFORMS.register()
@TRANSFORMS.register(name="label_to_tensor")
class LabelToTensor(BaseTransform):
    """``label`` (int / list[int]) -> torch.LongTensor。"""

    name = "label_to_tensor"

    def __call__(self, sample: Sample) -> Sample:
        if "label" in sample and not isinstance(sample["label"], torch.Tensor):
            sample["label"] = torch.as_tensor(sample["label"], dtype=torch.long)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="boxes_to_tensor")
class BoxesToTensor(BaseTransform):
    """``boxes`` (N, 4) -> torch.float32 Tensor；``labels`` -> torch.long。"""

    name = "boxes_to_tensor"

    def __call__(self, sample: Sample) -> Sample:
        if "boxes" in sample and not isinstance(sample["boxes"], torch.Tensor):
            sample["boxes"] = torch.as_tensor(sample["boxes"], dtype=torch.float32)
        if "labels" in sample and not isinstance(sample["labels"], torch.Tensor):
            sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.long)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="mask_to_tensor")
class MaskToTensor(BaseTransform):
    """``mask`` (H, W) numpy int64 -> torch.long Tensor。"""

    name = "mask_to_tensor"

    def __call__(self, sample: Sample) -> Sample:
        if "mask" in sample and not isinstance(sample["mask"], torch.Tensor):
            sample["mask"] = torch.as_tensor(sample["mask"], dtype=torch.long)
        return sample
