# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 头部子包：按任务组织（classification / segmentation / detection），每个头一个文...
"""头部子包：按任务组织（classification / segmentation / detection），每个头一个文件。

新增头：在对应任务子目录新建文件（继承 BaseModel + @HEADS.register()），并在
子包 __init__ 导出即可，互不影响。
"""

from .classification import (
    AdaFaceHead,
    ArcFaceHead,
    CircleLossHead,
    ClsHead,
    CosFaceHead,
    CurricularFaceHead,
    MultiLabelHead,
    NormFaceHead,
    PartialFCHead,
    SimCLRHead,
    SphereFaceHead,
)
from .detection import (
    CenterNetHead,
    DeformableDETRHead,
    DETRHead,
    DINODetrHead,
    FCOSHead,
    NMSFreeYOLOHead,
    RetinaNetHead,
    RTDETRFullHead,
    RTDETRHead,
    YOLOHead,
)
from .segmentation import (
    BiSeNetHead,
    DeepLabV3Head,
    DeepLabV3PlusHead,
    FCNHead,
    Mask2FormerHead,
    MaskFormerHead,
    PSPHead,
    SegFormerHead,
    SegFormerV2Head,
    SegFormerV3Head,
    SegHead,
    SwinUNetDecoder,
    UNetDecoder,
    UPerNetHead,
)

__all__ = [
    "ClsHead",
    "ArcFaceHead",
    "MultiLabelHead",
    "CosFaceHead",
    "CurricularFaceHead",
    "NormFaceHead",
    "PartialFCHead",
    "SimCLRHead",
    "SphereFaceHead",
    "CircleLossHead",
    "AdaFaceHead",
    "SegHead",
    "FCNHead",
    "DeepLabV3Head",
    "DeepLabV3PlusHead",
    "BiSeNetHead",
    "PSPHead",
    "UPerNetHead",
    "UNetDecoder",
    "SegFormerHead",
    "SegFormerV2Head",
    "SegFormerV3Head",
    "SwinUNetDecoder",
    "MaskFormerHead",
    "Mask2FormerHead",
    "FCOSHead",
    "RetinaNetHead",
    "YOLOHead",
    "RTDETRHead",
    "RTDETRFullHead",
    "DETRHead",
    "DINODetrHead",
    "CenterNetHead",
    "NMSFreeYOLOHead",
    "DeformableDETRHead",
]
