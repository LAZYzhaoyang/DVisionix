# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分类头子包。
"""分类头子包。"""

from .adaface import AdaFaceHead
from .arcface import ArcFaceHead
from .circle import CircleLossHead
from .cosface import CosFaceHead
from .curricularface import CurricularFaceHead
from .linear import ClsHead
from .multi_label import MultiLabelHead
from .normface import NormFaceHead
from .partial_fc import PartialFCHead
from .simclr import SimCLRHead
from .sphereface import SphereFaceHead

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
    "CircleLossHead",
    "SimCLRHead",
]
