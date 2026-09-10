# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DINO 风格检测头（compact：hybrid query selection + query denoising ...
"""DINO 风格检测头（compact：hybrid query selection + query denoising + box refinement）。"""

from typing import Optional

import torch
import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel
from ...layers import (
    DeformableDecoderLayer,
    DeformableEncoderLayer,
    DenoisingQueryGenerator,
    PositionEmbeddingSine,
    QuerySelection,
)


@HEADS.register()
@HEADS.register(name="dino_head")
class DINODetrHead(BaseModel):
    """DINO-lite：多尺度可变形编码器 -> 混合 query 选择 -> 解码器（可变形交叉注意力）+ box 细化。

    解码器逐层累积 box 更新（迭代细化），训练时输出各层中间框 ``intermediate_boxes``
    （配合 DINOLoss 的 look-forward-twice：第 i 层回归损失使用第 i+1 层的框）；
    推理仍只输出最后一层 logits/boxes，契约与 DETR 一致（可复用 detr_decode）。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        out_stride: Optional[int] = None,
        d_model: int = 256,
        num_queries: int = 300,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        num_points: int = 4,
        topk: int = 100,
        dropout: float = 0.1,
        dn_noise_scale_box: float = 0.2,
        in_channels=None,
        **kwargs,
    ):
        super().__init__(task_type="detection")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.out_stride = int(out_stride) if out_stride is not None else None
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_levels = len(in_channels_list)

        self.input_projs = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels_list])
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, d_model))
        self.encoder_layers = nn.ModuleList(
            [
                DeformableEncoderLayer(d_model, num_heads, self.num_levels, num_points, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        # 混合 query 选择
        self.sel_class_embed = nn.Linear(d_model, num_classes + 1)
        self.sel_box_embed = nn.Linear(d_model, 4)
        self.query_selection = QuerySelection(topk=topk)
        self.decoder_layers = nn.ModuleList(
            [
                DeformableDecoderLayer(d_model, num_heads, self.num_levels, num_points, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 4)
        )
        self.dn_generator = DenoisingQueryGenerator(
            d_model=d_model, num_classes=num_classes, noise_scale_box=dn_noise_scale_box
        )

    def _make_ref_points(self, shapes, device):
        refs = []
        for h, w in shapes:
            ys = (torch.arange(h, device=device) + 0.5) / h
            xs = (torch.arange(w, device=device) + 0.5) / w
            grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
            refs.append(grid)
        return torch.cat(refs, dim=0)

    def _resolve_image_hw(self, batch, feat0):
        """解析真实输入图像尺寸 ``(H, W)``，用于去噪目标的坐标归一化。

        v1.0.0 硬编码 ``feats[0].shape[2] * 4`` 作为图像尺寸。骨干步长不是 4 时该代理
        值就是错的：官方 ``configs/detection/dino_synthetic.yaml`` 用 3 个 stride=2 的
        stage，``feats[0]`` 实为 stride 2，代理值成为真值的 **2 倍**；而 ``DINOLoss``
        用的是真值，于是同一个 ``bbox_embed`` 在去噪分支与主分支被训练在**两个坐标系**
        里（CodePlan 7.1.2 D4）。

        优先级：
        1. ``batch["image"]`` 的真实尺寸 —— ``DetectionTask`` 训练时会把含 image 的
           batch 一并传入（``needs_batch = True``），这是正常路径；
        2. 显式配置的 ``out_stride``（第一个特征图相对输入的步长）推断；
        3. 都没有则**直接报错** —— 静默猜一个尺寸比报错危险得多。
        """
        image = batch.get("image") if isinstance(batch, dict) else None
        if image is not None and hasattr(image, "shape") and image.dim() >= 4:
            return (int(image.shape[-2]), int(image.shape[-1]))
        if self.out_stride is not None:
            return (int(feat0.shape[2]) * self.out_stride, int(feat0.shape[3]) * self.out_stride)
        raise ValueError(
            "DINODetrHead: 训练（含去噪）需要真实图像尺寸，但 batch 中缺少形如 (B, C, H, W) 的 "
            "'image'，且 head 未配置 out_stride。请传入含 image 的 batch，"
            "或在 head 配置里显式设置 out_stride（第一个特征图相对输入的步长）。"
        )

    def forward(self, feats, batch=None):
        """DINODetrHead 前向：多尺度特征(+batch) -> {logits, boxes}，训练含中间框/去噪项。"""
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        B = feats[0].shape[0]
        device = feats[0].device
        proj = [self.input_projs[i](f) for i, f in enumerate(feats)]
        shapes = [(p.shape[2], p.shape[3]) for p in proj]
        proj = [p + self.level_embed[i].view(1, -1, 1, 1) for i, p in enumerate(proj)]
        pos = [self.pos_embed(p) for p in proj]
        tokens = [
            p.flatten(2).permute(0, 2, 1) + pos[i].flatten(2).permute(0, 2, 1)
            for i, p in enumerate(proj)
        ]
        memory = torch.cat(tokens, dim=1)  # (B, T, d)
        ref = self._make_ref_points(shapes, device).unsqueeze(0).expand(B, -1, -1)
        for layer in self.encoder_layers:
            memory = layer(memory, proj, ref)

        sel_logits = self.sel_class_embed(memory)
        sel_box_raw = self.sel_box_embed(memory)
        queries, init_boxes, _ = self.query_selection(sel_logits, sel_box_raw, memory)
        k = queries.shape[1]
        query_pos = self.query_embed.weight[:k].unsqueeze(0).expand(B, -1, -1)
        tgt = queries
        ref_q = init_boxes[..., :2]  # cxcy 归一化

        training = self.training and batch is not None and batch.get("boxes") is not None
        if training:
            image_hw = self._resolve_image_hw(batch, feats[0])
            dn = self.dn_generator(batch["boxes"], batch["labels"], image_hw, device)
            dn_q, dn_cls_t, dn_box_t, dn_pos, dn_valid = dn
            dn_n = dn_q.shape[1]
            all_tgt = torch.cat([tgt, dn_q], dim=1)
            all_ref = torch.cat([ref_q, dn_box_t[..., :2]], dim=1)
            all_pos = torch.cat([query_pos, query_pos.new_zeros(B, dn_n, self.d_model)], dim=1)
        else:
            all_tgt, all_ref, all_pos = tgt, ref_q, query_pos

        # 逐层累积 box 更新（迭代细化）；最后一层即为最终框，训练/推理一致
        decoder_boxes = []
        box_acc = init_boxes  # (B, k, 4) 归一化 xywh anchor
        for layer in self.decoder_layers:
            all_tgt = layer(all_tgt, all_pos, proj, all_ref)
            delta = self.bbox_embed(all_tgt[:, :k])
            box_acc = box_acc + delta
            decoder_boxes.append(box_acc.sigmoid())

        main_tgt = all_tgt[:, :k]
        logits = self.class_embed(main_tgt)
        boxes = decoder_boxes[-1]
        out = {"logits": logits, "boxes": boxes}
        if training:
            # look-forward-twice 需要各层中间框（主分支）
            out["intermediate_boxes"] = decoder_boxes
        if training:
            dn_tgt = all_tgt[:, k:]
            dn_logits = self.class_embed(dn_tgt)
            dn_delta = self.bbox_embed(dn_tgt)
            dn_boxes = (dn_box_t + dn_delta).sigmoid()
            out.update(
                {
                    "dn_logits": dn_logits,
                    "dn_boxes": dn_boxes,
                    "dn_cls_target": dn_cls_t,
                    "dn_box_target": dn_box_t,
                    "dn_positive_mask": dn_pos,
                    "dn_valid": dn_valid,
                }
            )
        return out


__all__ = ["DINODetrHead"]
