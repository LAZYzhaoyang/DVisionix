# -*- coding: utf-8 -*-
"""分割组合模型：SegmentationModel（backbone + neck(可选) + 分割头）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..registry import BACKBONES, HEADS, MODELS, NECKS
from .base import BaseModel

# 多尺度头由 head 类属性 input_style="multi_scale" 自声明（装配器统一读取，无需硬编码名单）。


@MODELS.register()
@MODELS.register(name="segmentation_model")
class SegmentationModel(BaseModel):
    """骨干 + 分割头组合模型（可选 neck）。

    配置示例::

        model:
          type: segmentation_model
          num_classes: 21
          backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [1,2,3,4]}
          head: {type: deeplabv3_head, num_classes: 21}

    分割头支持：seg_head（1x1）/ fcn_head / deeplabv3_head / unet_decoder（需多尺度特征）。
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        upsample: bool = True,
        **kwargs,
    ):
        super().__init__(task_type="segmentation")
        bb_cfg = dict(backbone)
        bb_cfg.setdefault("features_only", True)
        if out_indices is not None:
            bb_cfg["out_indices"] = out_indices
        self.backbone = BACKBONES.build(bb_cfg)

        head_type = head.get("type") if isinstance(head, dict) else str(head)
        if neck is not None:
            neck_cfg = dict(neck)
            neck_cfg.setdefault("in_channels", self.backbone.out_channels)
            self.neck = NECKS.build(neck_cfg)
            neck_out = getattr(self.neck, "out_channels", None)
            self.in_channels = (
                neck_out
                if isinstance(neck_out, int)
                else (
                    neck_out[-1]
                    if isinstance(neck_out, (list, tuple))
                    else self.backbone.out_channels[-1]
                )
            )
        else:
            self.neck = None
            self.in_channels = self.backbone.out_channels[-1]

        head_cfg = dict(head)
        head_cls = (
            HEADS.get(head_type) if isinstance(head_type, str) and head_type in HEADS else None
        )
        is_multi = getattr(head_cls, "input_style", "single_scale") == "multi_scale"
        if is_multi:
            head_cfg.setdefault("in_channels_list", self._head_input_channels())
        else:
            head_cfg.setdefault("in_channels", self.in_channels)
        if "num_classes" not in head_cfg and num_classes is not None:
            head_cfg["num_classes"] = num_classes
        self.head = HEADS.build(head_cfg)
        self.num_classes = getattr(self.head, "num_classes", num_classes)
        self.upsample = upsample

    def extract_features(self, x: torch.Tensor):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def _head_input_channels(self):
        """多尺度头输入通道：有 neck 时取 neck 输出通道列表，否则取 backbone 多尺度。"""
        if self.neck is not None:
            neck_out = getattr(self.neck, "out_channels", None)
            if isinstance(neck_out, (list, tuple)):
                return list(neck_out)
            if neck_out is not None:
                num_outs = getattr(self.neck, "num_outs", None) or len(self.backbone.out_channels)
                return [neck_out] * int(num_outs)
        return list(self.backbone.out_channels)

    def decode(self, preds, image_hw, **kwargs):
        """委托给支持解码的 head（如 MaskFormerHead full 模式 -> masks/scores/labels）。"""
        decode_fn = getattr(self.head, "decode", None)
        if decode_fn is None:
            raise NotImplementedError(
                f"head {type(self.head).__name__} 不支持 decode()"
                "（仅 MaskFormerHead full 模式等可解码输出提供）"
            )
        return decode_fn(preds, image_hw, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feats = self.extract_features(x)
        if getattr(type(self.head), "input_style", "single_scale") == "multi_scale":
            out = self.head(feats)
        else:
            feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
            out = self.head(feat)
        if isinstance(out, dict):
            return out  # 多输出 head（如 MaskFormerHead full 模式）透传
        if self.upsample and out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


__all__ = ["SegmentationModel"]
