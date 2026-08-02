# -*- coding: utf-8 -*-
"""分割头子包。"""

from .deeplabv3 import DeepLabV3Head
from .deeplabv3plus import DeepLabV3PlusHead
from .fcn import FCNHead
from .maskformer import MaskFormerHead, maskformer_decode
from .psp import PSPHead
from .seg import SegHead
from .segformer import SegFormerHead
from .unet import UNetDecoder
from .upernet import UPerNetHead

__all__ = [
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
]
