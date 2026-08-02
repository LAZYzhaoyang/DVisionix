# -*- coding: utf-8 -*-
"""数据样本（Sample）契约。

整个 data 模块对样本字段名/类型/约定做了统一定义，所有 dataset / transform / collate
都遵循此契约，避免历史中 ``boxes`` vs ``bboxes``、``label`` vs ``labels`` 之类的混乱。

约定：
- ``image``：必填。numpy uint8 (H, W, C) BGR 或 RGB（详见 ImageMode），或文件路径 str，
  或已加载的 torch.Tensor (C, H, W)。
- ``label``：分类任务标签，int。
- ``boxes``：检测任务边界框，numpy float32 (N, 4) [x1, y1, x2, y2]（xyxy 绝对坐标）。
- ``labels``：检测任务类别，numpy int64 (N,)。
- ``mask``：分割任务掩码，numpy int64 (H, W)。
- ``meta``：可选 dict，存原始路径/尺寸/缩放比等元信息（dataset 内部透传，训练循环不消费）。
- 其余字段（如 ``keypoints``）可由自定义任务自由扩展，transform 应当按字段缺失则不处理。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List


class ImageMode(str, Enum):
    """图像色彩约定，dataset 输出与 transform 默认按此约定工作。"""

    RGB = "rgb"
    BGR = "bgr"


class Sample(dict):
    """样本字典——基于 ``dict`` 的轻量包装，便于 IDE 提示与字段校验。

    支持以 ``sample.image`` 或 ``sample["image"]`` 两种方式访问，``set/get`` 时会做
    字段名拼写检查（不在约定字段名列表中只警告，不抛错，方便自定义字段）。
    """

    _KNOWN_KEYS: List[str] = ["image", "label", "boxes", "labels", "mask", "meta"]

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_") or key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@dataclass
class ImageInfo:
    """图像相关元信息。"""

    height: int
    width: int
    channels: int = 3
    mode: ImageMode = ImageMode.RGB


@dataclass
class NormalizationSpec:
    """图像归一化参数（ImageNet 默认值）。"""

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    scale: float = 1.0 / 255.0  # 像素除以此值（默认 1/255 把 uint8 -> [0,1]）
