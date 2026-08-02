# -*- coding: utf-8 -*-
"""分割组合模型子包（每类一个文件）。"""

from .base import SegmentationModel
from .swin_unet import SwinUNet

__all__ = ["SegmentationModel", "SwinUNet"]
