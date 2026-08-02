# -*- coding: utf-8 -*-
"""头部子包：按任务组织（classification / segmentation / detection），每个头一个文件。

新增头：在对应任务子目录新建文件（继承 BaseModel + @HEADS.register()），并在
子包 __init__ 导出即可，互不影响。
"""

from .classification import ClsHead, ArcFaceHead, MultiLabelHead, CosFaceHead, SphereFaceHead, AdaFaceHead
from .segmentation import SegHead, FCNHead, DeepLabV3Head, UNetDecoder, SegFormerHead, MaskFormerHead
from .detection import DetHead, FCOSHead, RetinaNetHead, YOLOHead, DETRHead

__all__ = [
    "ClsHead", "ArcFaceHead", "MultiLabelHead", "CosFaceHead", "SphereFaceHead", "AdaFaceHead",
    "SegHead", "FCNHead", "DeepLabV3Head", "UNetDecoder", "SegFormerHead", "MaskFormerHead",
    "DetHead", "FCOSHead", "RetinaNetHead", "YOLOHead", "DETRHead",
]