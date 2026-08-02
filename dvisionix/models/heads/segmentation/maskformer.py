# -*- coding: utf-8 -*-
"""MaskFormer 风格分割头（query 掩码解码，compact 实现）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="maskformer_head")
class MaskFormerHead(BaseModel):
    """MaskFormer-lite：query 解码器 + 像素解码器，输出逐类语义 logits。

    像素解码器把多级特征上采样并融合；query 经 transformer 解码得到类别概率与掩码嵌入；
    语义 logits = sum_q P(query=c) * mask_q。可直接用 SegmentationTask 训练。

    注意：这是紧凑实现（静态 query，无 mask-attention 与匈牙利 mask 监督）；
    完整 Mask2Former（mask 分类损失 + 实例/全景评估）列为后续计划。
    """

    def __init__(
        self,
        in_channels_list,
        num_classes,
        d_model: int = 256,
        num_queries: int = 50,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        output_mode: str = "semantic",
    ):
        super().__init__(task_type="segmentation")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.output_mode = (
            output_mode  # "semantic"（张量）| "full"（dict，含 pred_logits/pred_masks）
        )

        self.pixel_decoder = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels_list])
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        target = feats[0].shape[-2:]
        pixel_feat = None
        for i, f in enumerate(feats):
            p = self.pixel_decoder[i](f)
            if p.shape[-2:] != target:
                p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            pixel_feat = p if pixel_feat is None else pixel_feat + p

        b, d, h, w = pixel_feat.shape
        memory = pixel_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, d)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = torch.zeros_like(queries)
        hs = self.decoder(tgt + queries, memory)  # (B, Q, d)

        class_logits = self.class_embed(hs)  # (B, Q, C+1)
        mask_embeds = self.mask_embed(hs)  # (B, Q, d)
        masks = torch.einsum("bqd,bdhw->bqhw", mask_embeds, pixel_feat)  # (B, Q, H, W)

        probs = torch.softmax(class_logits, dim=-1)[:, :, : self.num_classes]  # (B, Q, C)
        semantic = torch.einsum("bqc,bqhw->bchw", probs, masks)
        if self.output_mode == "full":
            return {"pred_logits": class_logits, "pred_masks": masks, "semantic_logits": semantic}
        return semantic


__all__ = ["MaskFormerHead", "maskformer_decode"]


def maskformer_decode(
    preds,
    image_hw,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    max_detections: int = 100,
):
    """MaskFormerHead full 模式解码：逐 query mask + 类别 + 分数。

    Returns:
        (masks_list, scores_list, labels_list)：每张图 (K, H, W) bool / (K,) / (K,)。
    """
    logits, masks = preds["pred_logits"], preds["pred_masks"]  # (B,Q,C+1), (B,Q,H,W)
    img_h, img_w = image_hw
    masks = masks.sigmoid()
    probs = torch.softmax(logits, dim=-1)[..., :-1]  # (B, Q, C)
    scores, labels = probs.max(dim=-1)
    masks_list, scores_list, labels_list = [], [], []
    for b in range(scores.shape[0]):
        keep = scores[b] >= score_threshold
        m = masks[b][keep] > mask_threshold
        s = scores[b][keep]
        lb = labels[b][keep]
        # 限制检测数
        if m.shape[0] > max_detections:
            _, idx = s.topk(max_detections)
            m, s, lb = m[idx], s[idx], lb[idx]
        masks_list.append(m)
        scores_list.append(s)
        labels_list.append(lb)
    return masks_list, scores_list, labels_list
