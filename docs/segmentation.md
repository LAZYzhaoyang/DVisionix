# 语义分割任务

分割任务复用统一的数据/模型/训练接口，只是数据字典用 `mask` 作为标签。

## 数据格式

分割样本遵循 `Sample` 协议：`image`（路径/numpy）+ `mask`（路径/numpy (H,W) 类别图），
并需要配套 transforms（`SegmentationTransforms` 会自动 resize + 归一化 + 转 tensor）：

```python
from dvisionix.data import CustomDataset
from dvisionix.data.transforms import SegmentationTransforms

samples = [
    {"image": "img1.png", "mask": "mask1.png"},
    {"image": "img2.png", "mask": "mask2.png"},
]
ds = CustomDataset(samples=samples, task_type="segmentation",
                   transforms=SegmentationTransforms(train=False, image_size=128))
```

`__getitem__` 输出：`{"image": Tensor(C, H, W), "mask": Tensor(H, W) long}`（类别索引）。

## 端到端训练

```bash
conda run -n dvisionix python tools/train.py --config configs/segmentation/demo_synthetic.yaml
```

由 `configs/segmentation/demo_synthetic.yaml` 驱动，使用合成的 image+mask，
流程：配置加载 -> 数据 -> `SimpleSegmentationModel` + `SegmentationTask` +
回调（ModelCheckpoint / EarlyStopping）-> 训练；验证日志输出 `mIoU / pixel_accuracy`。

## SegmentationTask 说明
- 默认损失：`CrossEntropy(ignore_index=255)`（忽略无效像素）；也可配置 `dice` / `combined_segmentation`。
- 默认指标：mIoU + pixel_accuracy。
- 优化器 / 调度器通过配置 `training.optimizer` / `training.scheduler` 调整。

## 真实数据
把配置里的 `data.dataset` 换成 `cityscapes` / `ade20k` 并提供真实 mask 路径即可；
注意训练时同一 batch 内图像/掩码尺寸需一致（用分割变换统一 resize）。