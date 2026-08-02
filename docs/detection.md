# 目标检测任务

DVisionix 提供**组件化检测器**（backbone → neck → head 即插即用），覆盖 anchor-free 与 anchor-based 两类主流范式，
并内置 COCO-style mAP 评估。教学级 `GridDetectionModel` 保留在 `models.toy` 供演示。

## 组件化检测器

| 检测器 | 类型 | 关键组件 | 注册名 |
| --- | --- | --- | --- |
| `FCOSDetector` | anchor-free | `FCOSHead` + `FCOSAssigner` + `FCOSDetectionLoss` | `fcos` |
| `RetinaNetDetector` | anchor-based | `RetinaNetHead` + `AnchorGenerator`（assigner 可选 `max_iou` / `atss`） | `retinanet` |
| `YOLODetector` | anchor-free（YOLOv8 风格） | `YOLOHead` + `TaskAlignedAssigner` + `YOLOLoss` | `yolo` |
| `DETRDetector` | transformer 端到端 | `DETRHead` + `HungarianMatcher` + `DETRLoss` | `detr` |
| `RTDETRDetector` | transformer 端到端（compact） | `RTDETRHead`（混合编码器 + query 选择）+ `DETRLoss` | `rtdetr` |
| `DeformableDETRDetector` | transformer 端到端（compact） | `DeformableDETRHead`（多尺度可变形注意力）+ `DETRLoss` | `deformable_detr` |

所有检测器由 `SingleStageDetector` 脚手架统一装配：backbone（自动 `features_only=True`）→ neck（可选 FPN / PANet）→ head。
YOLO 系列骨干可用 `csp_layer` / `elan_layer` 拼装（见 `configs/detection/yolov5_synthetic.yaml` / `yolov9_synthetic.yaml`）。
backbone / neck / head 均配置驱动，可自由组合（例如 `timm_backbone` 或 `SequentialBackbone` 配 `fpn` / `panet`）。

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
conda run -n dvisionix python tools/train.py --config configs/detection/fcos_synthetic.yaml
conda run -n dvisionix python tools/train.py --config configs/detection/retinanet_synthetic.yaml
conda run -n dvisionix python tools/train.py --config configs/detection/yolo_synthetic.yaml
```

以 `configs/detection/fcos_synthetic.yaml` 为例（其余配置结构相同，替换 `model.type` / `head.type` / `loss.type` 即可）：

```yaml
model:
  type: "fcos"
  num_classes: 3
  backbone: {type: "timm_backbone", name: "resnet18", pretrained: false, features_only: true, out_indices: [1, 2, 3, 4]}
  neck: {type: "fpn", out_channels: 64}
  head: {type: "fcos_head", num_classes: 3, strides: [4, 8, 16, 32]}

loss:
  type: "fcos_detection"     # retinanet_detection / yolo_detection / detr 等
  strides: [4, 8, 16, 32]
```

训练过程中验证日志会输出 `mAP / mAP_50 / mAP_75`（`evaluate_detection` + `DetectionMetrics`，COCO-style 101-point 插值）。

## 推理解码 + NMS

`forward` 只返回**原始预测**（多尺度 dict），推理侧解码由每个模型**自带的 `decode()` 方法**完成
（decode 与模型/head 写在同文件：如 `detectors/fcos.py` 的 `fcos_decode`、`detectors/base.py` 的 `detr_decode`、
`heads/segmentation/maskformer.py` 的 `maskformer_decode`）：

```python
import torch
from dvisionix.models import build_model

model = build_model({"type": "fcos", "num_classes": 3,
                     "backbone": {"type": "timm_backbone", "name": "resnet18", "pretrained": false,
                                  "features_only": true, "out_indices": [1, 2, 3, 4]},
                     "neck": {"type": "fpn", "out_channels": 64},
                     "head": {"type": "fcos_head", "num_classes": 3, "strides": [4, 8, 16, 32]}})
model.eval()
images = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    preds = model(images)                    # 原始预测（dict）
    boxes, scores, labels = model.decode(
        preds, image_hw=(256, 256),
        score_threshold=0.3,                 # 类别概率阈值
        iou_threshold=0.5,                   # NMS 的 IoU 阈值
        max_detections=100,
    )
# boxes[i]: Tensor(K, 4) [x1,y1,x2,y2] 像素坐标；scores[i]: Tensor(K,)；labels[i]: Tensor(K,)
```

decode 函数也可按需直接调用（顶层导出保持兼容）：

```python
from dvisionix.models import fcos_decode, retinanet_decode, yolo_decode, detr_decode, maskformer_decode
```

独立共享后处理原语（`dvisionix.models.postprocess`）：

```python
from dvisionix.models import nms, batched_nms, box_iou
keep = nms(boxes, scores, iou_threshold=0.5)                  # 单类
keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)  # 多类（类间不互相抑制）
```

## 组合性说明

- **backbone**：`timm_backbone`（ResNet 等任意 timm 模型，`pretrained` 可选）或 `sequential_backbone`（自拼层）。
- **neck**：`fpn` / `panet`（可选）；DETR 单尺度头可用可不用 neck；RT-DETR 亦可接 neck（按 neck 输出通道自动对齐）。
- **head**：`fcos_head` / `retinanet_head` / `yolo_head` / `detr_head` / `rtdetr_head` 均与任意 backbone / neck 组合；
  多尺度头（`input_style="multi_scale"` 自声明）自动注入 `in_channels_list`，单尺度头注入 `in_channels`，无需手动对齐通道数。

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
- 优化器 / 调度器 / 损失均可通过配置调整：`training.optimizer` / `training.scheduler` / `loss`。
- 检测损失自带 assigner（正负样本分配），无需手动配对。

## 真实数据与教学模型
- 真实数据集：`build_dataset({"type": "coco_detection", ...})`，或用 `CustomDataset` 提供你自己的标注。
- 教学级 `GridDetectionModel`（`models.toy`，骨干下采样 8 倍 + 网格单元单框预测）保留用于演示与快速验证，
  生产场景请使用上述组件化检测器。
