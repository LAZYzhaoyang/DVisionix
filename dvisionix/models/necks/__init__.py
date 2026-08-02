# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 颈部（neck）模块：多尺度特征融合。
"""颈部（neck）模块：多尺度特征融合。"""

from .fpn import FPN
from .panet import PANet
from .pixel_decoder import PixelDecoder

__all__ = ["FPN", "PANet", "PixelDecoder"]
