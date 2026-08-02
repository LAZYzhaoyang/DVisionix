# -*- coding: utf-8 -*-
"""分割头子包。"""

from .seg import SegHead
from .fcn import FCNHead
from .deeplabv3 import DeepLabV3Head
from .unet import UNetDecoder
from .segformer import SegFormerHead
from .maskformer import MaskFormerHead

__all__ = ["SegHead", "FCNHead", "DeepLabV3Head", "UNetDecoder", "SegFormerHead", "MaskFormerHead"]