# -*- coding: utf-8 -*-
"""分割头子包。"""

from .deeplabv3 import DeepLabV3Head
from .fcn import FCNHead
from .maskformer import MaskFormerHead
from .seg import SegHead
from .segformer import SegFormerHead
from .unet import UNetDecoder

__all__ = ["SegHead", "FCNHead", "DeepLabV3Head", "UNetDecoder", "SegFormerHead", "MaskFormerHead"]
