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
    CurricularFaceHead,
    MultiLabelHead,
    NormFaceHead,
    PartialFCHead,
    SphereFaceHead,
)
from .detection import (
    DeformableDETRHead,
    DetHead,
    DETRHead,
    FCOSHead,
    RetinaNetHead,
    RTDETRHead,
    YOLOHead,
)
from .segmentation import (
    DeepLabV3Head,
    DeepLabV3PlusHead,
    FCNHead,
    MaskFormerHead,
    PSPHead,
    SegFormerHead,
    SegHead,
    UNetDecoder,
    UPerNetHead,
    maskformer_decode,
)

__all__ = [
    "ClsHead",
    "ArcFaceHead",
    "MultiLabelHead",
    "CosFaceHead",
    "CurricularFaceHead",
    "NormFaceHead",
    "PartialFCHead",
    "SphereFaceHead",
    "AdaFaceHead",
    "SegHead",
    "FCNHead",
    "DeepLabV3Head",
    "DeepLabV3PlusHead",
    "PSPHead",
    "UPerNetHead",
    "UNetDecoder",
    "SegFormerHead",
    "MaskFormerHead",
    "maskformer_decode",
    "DetHead",
    "FCOSHead",
    "RetinaNetHead",
    "YOLOHead",
    "RTDETRHead",
    "DETRHead",
    "DeformableDETRHead",
]
