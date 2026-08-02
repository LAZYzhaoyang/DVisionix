# -*- coding: utf-8 -*-
"""头部子包：分类头 / 分割头 / 检测头。"""

from .cls_head import ClsHead, ArcFaceHead, MultiLabelHead
from .seg_head import SegHead, FCNHead, DeepLabV3Head, UNetDecoder
from .det_head import DetHead, FCOSHead, RetinaNetHead

__all__ = ["ClsHead", "ArcFaceHead", "MultiLabelHead", "SegHead", "FCNHead", "DeepLabV3Head", "UNetDecoder", "DetHead", "FCOSHead", "RetinaNetHead"]