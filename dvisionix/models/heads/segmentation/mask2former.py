# -*- coding: utf-8 -*-
"""Mask2Former 分割头（mask attention 解码器 + FPN 像素解码器，compact）。"""

import torch
import torch.nn as nn

from ....registry import HEADS, NECKS
from ...base import BaseModel
from ...postprocess import maskformer_decode


@HEADS.register()
@HEADS.register(name="mask2former_head")
class Mask2FormerHead(BaseModel):
    """Mask2Former-lite：FPN 像素解码器 + mask attention transformer 解码器。

    与 MaskFormerHead 的区别：
    - 解码器交叉注意力受**上一轮预测掩码**约束（mask attention），逐层细化掩码；
    - 每层都预测类别与掩码（iterative refinement），输出取最后一层。

    输出契约与 MaskFormerHead full 模式一致（pred_logits / pred_masks / semantic_logits），
    可复用 MaskFormerLoss（支持真实实例 GT）与 maskformer_decode / panoptic_decode。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        d_model: int = 256,
        num_queries: int = 50,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_mode: str = "full",
        pixel_decoder: dict = None,
        **kwargs,
    ):
        super().__init__(task_type="segmentation")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.output_mode = output_mode

        # 像素解码器：默认 necks.pixel_decoder（与 FPN/PANet 同级），可配置覆盖
        pd_cfg = dict(pixel_decoder) if pixel_decoder else {"type": "pixel_decoder"}
        pd_cfg.setdefault("in_channels", list(in_channels_list))
        pd_cfg.setdefault("d_model", d_model)
        self.pixel_decoder = NECKS.build(pd_cfg)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.level_query_embed = nn.Embedding(num_decoder_layers, d_model)
        self.decoder_layers = nn.ModuleList(
            [
                _MaskAttentionDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        feats_ms = self.pixel_decoder(feats)  # 多尺度，最精细在前
        pixel_feat = feats_ms[0]
        b, d, h, w = pixel_feat.shape
        memory = pixel_feat.flatten(2).permute(0, 2, 1)  # (B, HW, d)

        tgt = torch.zeros(b, self.num_queries, d, device=pixel_feat.device)
        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        mask_logits = None
        for layer_idx, layer in enumerate(self.decoder_layers):
            tgt = layer(
                tgt,
                query + self.level_query_embed.weight[layer_idx],
                memory,
                pixel_feat,
                mask_logits,
            )
            class_logits = self.class_embed(tgt)  # (B, Q, C+1)
            mask_embeds = self.mask_embed(tgt)
            mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embeds, pixel_feat)

        probs = torch.softmax(class_logits, dim=-1)[:, :, : self.num_classes]  # (B, Q, C)
        semantic = torch.einsum("bqc,bqhw->bchw", probs, mask_logits.sigmoid())
        if self.output_mode == "full":
            return {
                "pred_logits": class_logits,
                "pred_masks": mask_logits,
                "semantic_logits": semantic,
            }
        return semantic

    def decode(self, preds, image_hw, score_threshold=0.3, mask_threshold=0.5, max_detections=100):
        """full 模式推理解码，委托 maskformer_decode（契约一致）。"""
        return maskformer_decode(
            preds,
            image_hw,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            max_detections=max_detections,
        )


class _MaskAttentionDecoderLayer(nn.Module):
    """Mask2Former 解码层：自注意力 + mask 约束交叉注意力 + FFN。"""

    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = _MaskedCrossAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query, memory, pixel_feat, mask_logits):
        q = self.norm1(tgt + query)
        tgt = tgt + self.dropout(self.self_attn(q, q, q)[0])

        q2 = self.norm2(tgt + query)
        if mask_logits is not None:
            mask = (mask_logits.sigmoid() > 0.5).flatten(2).float()  # (B, Q, HW)
            attn_mask = torch.where(
                mask > 0, torch.zeros_like(mask), torch.full_like(mask, float("-inf"))
            )
        else:
            attn_mask = None
        tgt = tgt + self.dropout(self.cross_attn(q2, memory, memory, attn_mask=attn_mask))
        tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt


class _MaskedCrossAttention(nn.Module):
    """手写多头交叉注意力，支持逐 query 的注意力掩码（batch 内每个 query 独立掩码）。"""

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        B, N, d = query.shape
        L = key.shape[1]
        H = self.num_heads
        hd = d // H
        q = self.q_proj(query).reshape(B, N, H, hd).permute(0, 2, 1, 3)  # (B,H,N,hd)
        k = self.k_proj(key).reshape(B, L, H, hd).permute(0, 2, 3, 1)  # (B,H,hd,L)
        v = self.v_proj(value).reshape(B, L, H, hd).permute(0, 2, 1, 3)  # (B,H,L,hd)
        scores = q @ k / (hd**0.5)  # (B,H,N,L)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)
        attn = scores.softmax(dim=-1)
        out = attn @ v  # (B,H,N,hd)
        out = out.permute(0, 2, 1, 3).reshape(B, N, d)
        return self.out_proj(out)


__all__ = ["Mask2FormerHead"]
