# -*- coding: utf-8 -*-
"""头部（head）模块：分类 / 分割 / 检测头。"""

from .cls_head import ClsHead
from .seg_head import SegHead
from .det_head import DetHead

__all__ = ["ClsHead", "SegHead", "DetHead"]
