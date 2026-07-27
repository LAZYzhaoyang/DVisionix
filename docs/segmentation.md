# 语义分割任务

分割任务复用统一的数据/模型/训练接口，只是数据字典用 `mask` 作为标签。

## 数据格式

分割样本需要 `image_path` 和 `mask_path`；`__getitem__` 输出：

```python
{"image": Tensor(C, H, W), "mask": Tensor(H, W)}  # mask 为类别索引(long)
```

用 `CustomDataset` 构造：

```python
from dvisionix.data import CustomDataset

samples = [
    {"image_path": "img1.png", "mask_path": "mask1.png"},
    {"image_path": "img2.png", "mask_path": "mask2.png"},
]
ds = CustomDataset(task_type="segmentation", samples=samples, num_classes=3)
```

## 端到端训练

```bash
conda run -n dvisionix python demos/train_segmentation.py
```

该 demo 由 `configs/segmentation/demo_synthetic.yaml` 驱动，使用合成的
image+mask（固定尺寸），流程：配置加载 -> 数据 -> `SimpleSegmentationModel` +
`SegmentationTask` + 回调(ModelCheckpoint/TensorBoard/EarlyStopping) -> 训练。

## SegmentationTask 说明
- 损失：`CrossEntropyLoss(ignore_index=255)`，忽略无效像素。
- 指标：像素准确率（忽略 `ignore_index`）。
- 优化器：AdamW + CosineAnnealingLR。

## 真实数据
把配置里的 `data.dataset` 换成 `cityscapes`/`ade20k` 并提供真实 `mask_path`
即可；注意训练时同一 batch 内图像/掩码尺寸需一致（可用分割变换统一 resize）。
