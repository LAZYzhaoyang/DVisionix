# -*- coding: utf-8 -*-
"""分类头子包。"""

from .linear import ClsHead
from .arcface import ArcFaceHead
from .multi_label import MultiLabelHead
from .cosface import CosFaceHead
from .sphereface import SphereFaceHead
from .adaface import AdaFaceHead

__all__ = ["ClsHead", "ArcFaceHead", "MultiLabelHead", "CosFaceHead", "SphereFaceHead", "AdaFaceHead"]