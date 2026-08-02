# -*- coding: utf-8 -*-
"""分割头子包。"""

from .bisenet import BiSeNetHead
from .deeplabv3 import DeepLabV3Head
from .deeplabv3plus import DeepLabV3PlusHead
from .fcn import FCNHead
from .mask2former import Mask2FormerHead
from .maskformer import MaskFormerHead
from .psp import PSPHead
from .seg import SegHead
from .segformer import SegFormerHead
from .segformer_v2 import SegFormerV2Head
from .swin_unet import SwinUNetDecoder
from .unet import UNetDecoder
from .upernet import UPerNetHead

__all__ = [
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
    "SwinUNetDecoder",
    "MaskFormerHead",
    "Mask2FormerHead",
]
