# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 标签字段的张量化与基础转换。
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
    """``mask`` -> ``torch.long`` Tensor，并校验形状与取值。

    分割标签必须满足 ``CrossEntropyLoss`` 的 long 契约。v1.0.0 只在输入**不是**
    Tensor 时才转换，已经是 float Tensor 的 mask 会被原样保留并在 loss 处报错
    （CodePlan 7.1.2 D10）。现在无论输入类型一律强制 long，并在转换前拦截
    两类会静默出错的情况：

    - 形状不是 ``(H, W)`` 或 ``(H, W, C)``；
    - 取值含负值（负标签会让 ``CrossEntropyLoss`` 报错或与 ``ignore_index`` 混淆）；
    - 浮点输入含非整数值（``.long()`` 会静默截断）。
    """

    name = "mask_to_tensor"

    def __call__(self, sample: Sample) -> Sample:
        if "mask" not in sample:
            return sample

        raw = torch.as_tensor(sample["mask"])
        if raw.is_floating_point() and raw.numel() and not torch.allclose(raw, raw.round()):
            raise ValueError(
                "MaskToTensor: mask 为浮点且含非整数值，转 long 会静默截断；"
                "请确认分割标签是整数类别图"
            )

        tensor = raw.long()
        if tensor.ndim not in (2, 3):
            raise ValueError(
                f"MaskToTensor: mask 形状必须是 (H, W) 或 (H, W, C)，"
                f"当前为 {tuple(tensor.shape)}"
            )
        if tensor.numel() and int(tensor.min()) < 0:
            raise ValueError(
                f"MaskToTensor: mask 含负值（min={int(tensor.min())}）；" f"类别值必须为非负整数"
            )

        sample["mask"] = tensor
        return sample
