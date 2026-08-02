# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分割组合模型子包（每类一个文件）。
"""分割组合模型子包（每类一个文件）。"""

from .base import SegmentationModel
from .swin_unet import SwinUNet

__all__ = ["SegmentationModel", "SwinUNet"]
