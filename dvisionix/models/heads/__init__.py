# -*- coding: utf-8 -*-
"""头部子包：按任务组织（classification / segmentation / detection），每个头一个文件。

新增头：在对应任务子目录新建文件（继承 BaseModel + @HEADS.register()），并在
子包 __init__ 导出即可，互不影响。
"""

from .classification import (
    AdaFaceHead,
    ArcFaceHead,
    ClsHead,
    CosFaceHead,
    MultiLabelHead,
    SphereFaceHead,
)
from .detection import DetHead, DETRHead, FCOSHead, RetinaNetHead, RTDETRHead, YOLOHead
from .segmentation import (
    DeepLabV3Head,
    FCNHead,
    MaskFormerHead,
    SegFormerHead,
    SegHead,
    UNetDecoder,
)

__all__ = [
    "ClsHead",
    "ArcFaceHead",
    "MultiLabelHead",
    "CosFaceHead",
    "SphereFaceHead",
    "AdaFaceHead",
    "SegHead",
    "FCNHead",
    "DeepLabV3Head",
    "UNetDecoder",
    "SegFormerHead",
    "MaskFormerHead",
    "DetHead",
    "FCOSHead",
    "RetinaNetHead",
    "YOLOHead", "RTDETRHead",
    "DETRHead",
]
