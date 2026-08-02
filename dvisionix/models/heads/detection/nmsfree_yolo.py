# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: YOLOv10 风格检测头（NMS-free，结构同 YOLOHead，配合 one-to-one 损失训练）。
"""YOLOv10 风格检测头（NMS-free，结构同 YOLOHead，配合 one-to-one 损失训练）。"""

from ....registry import HEADS
from .yolo import YOLOHead


@HEADS.register()
@HEADS.register(name="yolo_v10_head")
class NMSFreeYOLOHead(YOLOHead):
    """YOLOv10 风格解耦头：结构与 YOLOHead 一致，但训练配合 one-to-one 匹配损失
    （yolo_v10_detection），推理时无需 NMS（直接 top-k）。

    输入：FPN/PANet 特征列表；输出 dict：cls 每层 (B, C, H, W)、reg 每层 (B, 4, H, W)。
    """

    pass


__all__ = ["NMSFreeYOLOHead"]
