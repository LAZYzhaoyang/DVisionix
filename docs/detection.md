# 目标检测任务

DVisionix 内置一个自洽、可端到端训练**并可评估 mAP** 的单阶段网格检测器
（YOLO 风格），用于演示与教学，无需依赖外部检测框架。

## 组成
- `GridDetectionModel`：骨干下采样 8 倍，输出网格张量 `(B, 5 + num_classes, GH, GW)`，
  每个网格单元预测 `objectness(1) + box(4: cx,cy,w,h) + 类别(num_classes)`。
  并提供 `decode(...)` 方法完成推理侧解码 + NMS。
- `DetectionTask`：配合 `GridDetectionLoss`（内部含 `GridAssigner` 中心点分配），
  损失 = objectness BCE（全网格）+ box L1（正样本）+ 分类 CE（正样本）。
- `detection_collate`：处理变长 `boxes/labels`，DataLoader 必须使用它。
- `evaluate_detection`：解码 + NMS + COCO-style mAP 评估的一站式工具。

## 数据格式

检测样本遵循 `Sample` 协议：`image`（路径/numpy）+ `boxes`（float32 (N,4) xyxy）+ `labels`（int64 (N,)），
并需要配套 transforms（至少 `BoxSyncResize` + `ToTensor`）与 `detection_collate`：

```python
from dvisionix.data import CustomDataset, detection_collate
from dvisionix.data.transforms import DetectionTransforms
from torch.utils.data import DataLoader
import numpy as np

samples = [
    {"image": "img1.png", "boxes": np.array([[5, 5, 25, 25]], dtype=np.float32), "labels": np.array([0])},
    {"image": "img2.png", "boxes": np.array([[3, 3, 30, 30], [40, 40, 60, 60]], dtype=np.float32), "labels": np.array([1, 2])},
]
ds = CustomDataset(samples=samples, task_type="detection",
                   transforms=DetectionTransforms(train=False, image_size=64))
loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=detection_collate)
```

`__getitem__` 输出：`{"image": Tensor(C,H,W), "boxes": Tensor(N,4), "labels": Tensor(N,)}`。

## 端到端训练 + 评估

```bash
conda run -n dvisionix python tools/train.py --config configs/detection/demo_synthetic.yaml
```

由 `configs/detection/demo_synthetic.yaml` 驱动，使用**可学习的**合成数据
（按类别绘制实心矩形）。训练过程中验证日志会输出 `mAP / mAP_50 / mAP_75`。

## 推理解码 + NMS

```python
import torch
from dvisionix.models import GridDetectionModel

model = GridDetectionModel(num_classes=3)
images = torch.randn(2, 3, 64, 64)
preds = model(images)                                  # (B, 5+num_classes, GH, GW)
boxes, scores, labels = model.decode(
    preds, image_hw=(64, 64),
    score_threshold=0.3,      # objectness * 类别概率 的阈值
    iou_threshold=0.5,        # NMS 的 IoU 阈值
    max_detections=100,
)
# boxes[i]: Tensor(K, 4) [x1,y1,x2,y2] 像素坐标；scores[i]: Tensor(K,)；labels[i]: Tensor(K,)
```

独立 NMS 工具：

```python
from dvisionix.models import nms, batched_nms, box_iou
keep = nms(boxes, scores, iou_threshold=0.5)                  # 单类
keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)  # 多类（类间不互相抑制）
```

## mAP 评估

```python
import torch
from torch.utils.data import DataLoader
from dvisionix.data import detection_collate
from dvisionix.training import evaluate_detection

loader = DataLoader(val_ds, batch_size=8, collate_fn=detection_collate)
metrics = evaluate_detection(
    model, loader, num_classes=3, device=torch.device("cpu"),
    score_threshold=0.3, iou_threshold=0.5,
)
print(metrics)   # {"mAP": ..., "mAP_50": ..., "mAP_75": ...}
```

底层为 `dvisionix.metrics.DetectionMetrics`（COCO-style，101-point 插值）。
正确性已用单元测试保证：完美预测 mAP≈1.0，完全错误预测 mAP=0.0。

## 训练信息
- 优化器 / 调度器 / 损失均可通过配置调整：
  `training.optimizer` / `training.scheduler` / `loss`（如 `grid_detection` 的 `obj_weight/box_weight/cls_weight`）。
- 训练/验证日志输出 `obj_loss / box_loss / cls_loss / cls_acc`，验证另含 `mAP / mAP_50 / mAP_75`。

## 真实数据与进阶
- 真实数据集：`build_dataset({"type": "coco_detection", ...})`，
  或用 `CustomDataset` 提供你自己的标注。
- 该检测器为教学级实现（单元格单框、无多尺度/anchor）。若在合成数据上追求高 mAP，
  可增大训练轮数使框回归充分收敛；生产级检测建议接入成熟框架，
  或扩展 `GridDetectionModel` 的多框预测与更强的目标分配策略。