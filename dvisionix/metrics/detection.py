# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\metrics\\detection.py

"""
检测任务指标

计算 COCO-style mAP (mean Average Precision) 等检测指标。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class DetectionMetrics:
    """检测指标计算器（COCO-style）"""
    
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[List[float]] = None,
        recall_thresholds: Optional[List[float]] = None,
    ):
        """
        Args:
            num_classes: 类别数量
            iou_thresholds: IoU 阈值列表，默认 [0.5, 0.55, ..., 0.95]
            recall_thresholds: Recall 阈值列表，默认 [0, 0.01, ..., 1.0]
        """
        self.num_classes = num_classes
        
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.5, 0.95, 10).tolist()
        else:
            self.iou_thresholds = iou_thresholds
        
        if recall_thresholds is None:
            self.recall_thresholds = np.linspace(0, 1.0, 101).tolist()
        else:
            self.recall_thresholds = recall_thresholds
        
        self.reset()
    
    def reset(self) -> None:
        """重置统计"""
        # 按类别存储预测和目标
        self.detections: Dict[int, List[Dict]] = {
            c: [] for c in range(self.num_classes)
        }
        self.annotations: Dict[int, List[Dict]] = {
            c: [] for c in range(self.num_classes)
        }
        self.image_ids: set = set()
    
    def update(
        self,
        pred_boxes: List[torch.Tensor],
        pred_scores: List[torch.Tensor],
        pred_labels: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> None:
        """
        更新统计
        
        Args:
            pred_boxes: 预测边界框列表，每个元素 (N, 4) [x1, y1, x2, y2]
            pred_scores: 预测置信度列表，每个元素 (N,)
            pred_labels: 预测类别列表，每个元素 (N,)
            target_boxes: 目标边界框列表，每个元素 (M, 4)
            target_labels: 目标类别列表，每个元素 (M,)
        """
        batch_size = len(pred_boxes)
        
        for i in range(batch_size):
            image_id = len(self.image_ids)
            self.image_ids.add(image_id)
            
            # 存储检测结果
            boxes_np = pred_boxes[i].cpu().numpy()
            scores_np = pred_scores[i].cpu().numpy()
            labels_np = pred_labels[i].cpu().numpy()
            
            for box, score, label in zip(boxes_np, scores_np, labels_np):
                label_int = int(label)
                if 0 <= label_int < self.num_classes:
                    self.detections[label_int].append({
                        "image_id": image_id,
                        "box": box,
                        "score": score,
                    })
            
            # 存储标注
            t_boxes_np = target_boxes[i].cpu().numpy()
            t_labels_np = target_labels[i].cpu().numpy()
            
            for box, label in zip(t_boxes_np, t_labels_np):
                label_int = int(label)
                if 0 <= label_int < self.num_classes:
                    self.annotations[label_int].append({
                        "image_id": image_id,
                        "box": box,
                        "matched": False,  # 匹配标记
                    })
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含各指标的字典
        """
        aps_per_class = []
        
        for c in range(self.num_classes):
            if len(self.annotations[c]) == 0:
                # 没有该类别的标注
                continue
            
            # 按分数排序检测结果
            dets = sorted(self.detections[c], key=lambda x: x["score"], reverse=True)
            
            # 按图像分组标注
            anns_by_image: Dict[int, List[Dict]] = {}
            for ann in self.annotations[c]:
                img_id = ann["image_id"]
                if img_id not in anns_by_image:
                    anns_by_image[img_id] = []
                anns_by_image[img_id].append({"box": ann["box"], "matched": False})
            
            if len(dets) == 0:
                aps_per_class.append(0.0)
                continue
            
            # 计算每个 IoU 阈值的 AP
            aps_iou = []
            
            for iou_thresh in self.iou_thresholds:
                tp = np.zeros(len(dets))
                fp = np.zeros(len(dets))
                
                # 重置匹配标记
                for img_anns in anns_by_image.values():
                    for ann in img_anns:
                        ann["matched"] = False
                
                for det_idx, det in enumerate(dets):
                    img_id = det["image_id"]
                    det_box = det["box"]
                    
                    if img_id not in anns_by_image:
                        fp[det_idx] = 1
                        continue
                    
                    # 找最佳匹配
                    best_iou = -1
                    best_ann_idx = -1
                    
                    for ann_idx, ann in enumerate(anns_by_image[img_id]):
                        if ann["matched"]:
                            continue
                        
                        iou = self._compute_iou(det_box, ann["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_ann_idx = ann_idx
                    
                    if best_iou >= iou_thresh:
                        tp[det_idx] = 1
                        anns_by_image[img_id][best_ann_idx]["matched"] = True
                    else:
                        fp[det_idx] = 1
                
                # 计算累计 TP 和 FP
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                # 召回率和精确率
                num_anns = sum(len(anns) for anns in anns_by_image.values())
                recall = tp_cumsum / num_anns if num_anns > 0 else tp_cumsum
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                
                # 计算 AP（101-point interpolation）
                ap = self._compute_ap(recall, precision)
                aps_iou.append(ap)
            
            aps_per_class.append(np.mean(aps_iou))
        
        result = {
            "mAP": np.mean(aps_per_class) if aps_per_class else 0.0,
            "mAP_50": self._compute_map_at_iou(0.5),
            "mAP_75": self._compute_map_at_iou(0.75),
        }
        
        return result
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection + 1e-8
        
        return intersection / union
    
    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """使用 101-point interpolation 计算 AP"""
        ap = 0.0
        
        for r_thresh in self.recall_thresholds:
            # 找到所有 recall >= r_thresh 的 precision
            mask = recall >= r_thresh
            if np.any(mask):
                p = np.max(precision[mask])
            else:
                p = 0.0
            ap += p
        
        return ap / len(self.recall_thresholds)
    
    def _compute_map_at_iou(self, iou_thresh: float) -> float:
        """计算特定 IoU 阈值下的 mAP"""
        # （简化实现，实际应该缓存中间结果）
        aps_per_class = []
        
        for c in range(self.num_classes):
            if len(self.annotations[c]) == 0:
                continue
            
            dets = sorted(self.detections[c], key=lambda x: x["score"], reverse=True)
            
            anns_by_image: Dict[int, List[Dict]] = {}
            for ann in self.annotations[c]:
                img_id = ann["image_id"]
                if img_id not in anns_by_image:
                    anns_by_image[img_id] = []
                anns_by_image[img_id].append({"box": ann["box"], "matched": False})
            
            if len(dets) == 0:
                aps_per_class.append(0.0)
                continue
            
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))
            
            for img_anns in anns_by_image.values():
                for ann in img_anns:
                    ann["matched"] = False
            
            for det_idx, det in enumerate(dets):
                img_id = det["image_id"]
                det_box = det["box"]
                
                if img_id not in anns_by_image:
                    fp[det_idx] = 1
                    continue
                
                best_iou = -1
                best_ann_idx = -1
                
                for ann_idx, ann in enumerate(anns_by_image[img_id]):
                    if ann["matched"]:
                        continue
                    
                    iou = self._compute_iou(det_box, ann["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_ann_idx = ann_idx
                
                if best_iou >= iou_thresh:
                    tp[det_idx] = 1
                    anns_by_image[img_id][best_ann_idx]["matched"] = True
                else:
                    fp[det_idx] = 1
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            num_anns = sum(len(anns) for anns in anns_by_image.values())
            recall = tp_cumsum / num_anns if num_anns > 0 else tp_cumsum
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
            ap = self._compute_ap(recall, precision)
            aps_per_class.append(ap)
        
        return np.mean(aps_per_class) if aps_per_class else 0.0
