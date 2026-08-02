# -*- coding: utf-8 -*-
"""分类头子包。"""

from .adaface import AdaFaceHead
from .arcface import ArcFaceHead
from .cosface import CosFaceHead
from .curricularface import CurricularFaceHead
from .linear import ClsHead
from .multi_label import MultiLabelHead
from .normface import NormFaceHead
from .partial_fc import PartialFCHead
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
]
