# -*- coding: utf-8 -*-
"""组合式模型：backbone (+ neck) + head。

通过配置拼装任意骨干、颈部与头部，体现 backbone -> neck -> head 的组件化架构。
后处理（decode/NMS）不在此处，保持模型只输出原始预测。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ...registry import MODELS, BACKBONES, NECKS, HEADS


@MODELS.register()
@MODELS.register(name="generalized")
class GeneralizedModel(BaseModel):
    """通用组合模型。

    配置示例（分类）::

        model:
          type: generalized
          task_type: classification
          backbone: {type: timm_backbone, name: resnet18, pretrained: false}
          head: {type: cls_head, num_classes: 10, dropout: 0.1}

    配置示例（分割）::

        model:
          type: generalized
          task_type: segmentation
          backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [4]}
          head: {type: seg_head, num_classes: 21, output_size: [128, 128]}

    配置示例（检测）::

        model:
          type: generalized
          task_type: detection
          backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [1,2,3,4]}
          neck: {type: fpn, out_channels: 256}
          head: {type: det_head, num_classes: 3}
    """

    def __init__(self, task_type, backbone, head, neck=None):
        super().__init__(task_type=task_type)
        self.task_type = task_type
        self.use_neck = neck is not None

        bb_cfg = dict(backbone)
        if task_type != "classification":
            bb_cfg.setdefault("features_only", True)
        self.backbone = BACKBONES.build(bb_cfg)

        if self.use_neck:
            neck_cfg = dict(neck)
            neck_cfg.setdefault("in_channels", self.backbone.out_channels)
            self.neck = NECKS.build(neck_cfg)
            head_in = getattr(self.neck, "out_channels", None)
            if isinstance(head_in, list):
                head_in = head_in[-1]
        else:
            self.neck = None
            if task_type == "classification":
                head_in = self.backbone.num_features
            else:
                head_in = self.backbone.out_channels[-1]

        head_cfg = dict(head)
        head_cfg["in_channels"] = head_in
        self.head = HEADS.build(head_cfg)
        self.num_classes = head_cfg.get("num_classes")

    def forward(self, x, **kwargs):
        feats = self.backbone(x)
        if self.task_type == "classification":
            feat = feats
            if self.use_neck:
                feat = self.neck(feats)[-1]
            return self.head(feat)

        if self.use_neck:
            feats = self.neck(feats)
        feat = feats[-1]
        out = self.head(feat)

        if self.task_type == "segmentation" and out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
