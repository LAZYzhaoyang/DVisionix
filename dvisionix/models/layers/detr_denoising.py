# -*- coding: utf-8 -*-
"""DINO query denoising（去噪训练 query 生成，含对比正负样本）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .attention import PositionEmbeddingSine


@LAYERS.register()
@LAYERS.register(name="denoising_query_generator")
class DenoisingQueryGenerator(nn.Module):
    """对每个 GT 生成正/负噪声 query（compact：每 GT 各 1 个正、1 个负样本）。

    - 正样本：框加小噪声、类别小概率错；参与分类 + 回归损失。
    - 负样本：框加大噪声、类别必错；只参与分类损失（回归损失被 mask）。
    返回 (dn_queries, dn_logits_target, dn_boxes_target, positive_mask, dn_valid)。
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        noise_scale_box: float = 0.2,
        noise_scale_label: float = 0.5,
        label_noise_prob: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.noise_scale_box = float(noise_scale_box)
        self.noise_scale_label = float(noise_scale_label)
        self.label_noise_prob = float(label_noise_prob)
        self.label_embed = nn.Embedding(num_classes + 1, d_model)
        self.content_embed = nn.Linear(4, d_model)
        self.pos_embed = PositionEmbeddingSine(d_model // 2, normalize=False)

    def forward(self, gt_boxes, gt_labels, image_hw, device):
        """gt_boxes: List[(M,4)] xyxy 像素；gt_labels: List[(M,)]。返回批量张量。"""
        B = len(gt_boxes)
        dn_queries, dn_cls_tgt, dn_box_tgt, pos_mask, valid = [], [], [], [], []
        img_h, img_w = image_hw
        for b in range(B):
            boxes = gt_boxes[b].to(device).float()
            labels = gt_labels[b].to(device).long()
            m = boxes.shape[0]
            if m == 0:
                dn_queries.append(torch.zeros(0, self.d_model, device=device))
                dn_cls_tgt.append(torch.zeros(0, dtype=torch.long, device=device))
                dn_box_tgt.append(torch.zeros(0, 4, device=device))
                pos_mask.append(torch.zeros(0, dtype=torch.bool, device=device))
                valid.append(torch.zeros(0, dtype=torch.bool, device=device))
                continue
            # 归一化框 cxcywh
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h
            w = (boxes[:, 2] - boxes[:, 0]) / img_w
            h = (boxes[:, 3] - boxes[:, 1]) / img_h
            norm = torch.stack([cx, cy, w, h], dim=1)  # (M, 4)
            # 正样本：小噪声
            pos_noise = torch.randn_like(norm) * self.noise_scale_box
            pos_box = (norm + pos_noise).clamp(0, 1)
            # 负样本：大噪声（坐标扰动更大）
            neg_noise = torch.randn_like(norm) * (self.noise_scale_box * 3)
            neg_box = (norm + neg_noise).clamp(0, 1)
            pos_label = labels.clone()
            # 正样本类别小概率错
            if self.label_noise_prob > 0:
                flip = torch.rand(m, device=device) < self.label_noise_prob
                if flip.any():
                    pos_label[flip] = torch.randint(
                        0, self.num_classes, (int(flip.sum()),), device=device
                    )
            # 负样本类别必错（随机错类）
            neg_label = torch.randint(0, self.num_classes, (m,), device=device)
            dn_box = torch.cat([pos_box, neg_box], dim=0)  # (2M, 4)
            dn_lbl = torch.cat([pos_label, neg_label], dim=0)
            p_mask = torch.cat(
                [
                    torch.ones(m, dtype=torch.bool, device=device),
                    torch.zeros(m, dtype=torch.bool, device=device),
                ]
            )
            v_mask = torch.ones(2 * m, dtype=torch.bool, device=device)
            # content: 噪声框 -> 内容特征（线性）
            content = self.content_embed(dn_box)
            # 位置编码（对噪声框的归一化坐标做正弦编码，近似）
            phw = torch.zeros(2 * m, self.d_model, device=device)
            # queries = label_embed + content + pos
            queries = self.label_embed(dn_lbl) + content + phw
            dn_queries.append(queries)
            dn_cls_tgt.append(dn_lbl)
            dn_box_tgt.append(dn_box)
            pos_mask.append(p_mask)
            valid.append(v_mask)
        # 对齐到 batch 最大长度
        max_n = max(v.shape[0] for v in valid) if valid else 0
        qs, cls_t, box_t, pm, vm = [], [], [], [], []
        for b in range(B):
            n = valid[b].shape[0]
            pad = max_n - n
            if pad > 0:
                qs.append(
                    torch.cat([dn_queries[b], torch.zeros(pad, self.d_model, device=device)], 0)
                )
                cls_t.append(
                    torch.cat([dn_cls_tgt[b], torch.zeros(pad, dtype=torch.long, device=device)], 0)
                )
                box_t.append(torch.cat([dn_box_tgt[b], torch.zeros(pad, 4, device=device)], 0))
                pm.append(
                    torch.cat([pos_mask[b], torch.zeros(pad, dtype=torch.bool, device=device)], 0)
                )
                vm.append(
                    torch.cat([valid[b], torch.zeros(pad, dtype=torch.bool, device=device)], 0)
                )
            else:
                qs.append(dn_queries[b])
                cls_t.append(dn_cls_tgt[b])
                box_t.append(dn_box_tgt[b])
                pm.append(pos_mask[b])
                vm.append(valid[b])
        return (
            torch.stack(qs),
            torch.stack(cls_t),
            torch.stack(box_t),
            torch.stack(pm),
            torch.stack(vm),
        )


__all__ = ["DenoisingQueryGenerator"]
