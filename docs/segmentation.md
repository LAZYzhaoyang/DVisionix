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
流程：配置加载 -> 数据 -> `SegmentationModel`（backbone + 分割头）+ `SegmentationTask` +
回调（ModelCheckpoint / EarlyStopping）-> 训练；验证日志输出 `mIoU / pixel_accuracy`。

## 分割头

组件化分割模型 `SegmentationModel`（注册名 `segmentation_model`）把 backbone 与分割头即插即用组合：

- 单尺度头（自动注入 `in_channels`）：`seg_head`（1x1 卷积）/ `fcn_head` / `deeplabv3_head`（ASPP）/ `psp_head`（金字塔池化）。
- 多尺度头（自动注入 `in_channels_list`）：`unet_decoder` / `segformer_head`（MLP 解码）/ `maskformer_head`（query 掩码解码）/ `mask2former_head`（mask attention 解码）/ `upernet_head`（FPN+PPM）/ `deeplabv3plus_head`（ASPP+低层解码）。
- `MaskFormerHead` 支持 `output_mode="full"`：返回 `pred_logits / pred_masks / semantic_logits`，
  配合 `MaskFormerLoss`（匈牙利 mask 匹配）做实例级 mask 预测；推理解码可用 `model.decode(preds, image_hw)`（内部委托 `maskformer_decode`）。
  实例分割训练用 `MaskFormerTask`（loss=MaskFormerLoss，指标=mask mAP）；`MaskFormerTask(panoptic=True)` 时验证日志额外输出
  `PQ / SQ / RQ`（GT 取 `batch["panoptic"]`，缺省退化语义）；`MaskFormerLoss` 支持真实实例 GT（`instance_masks` / `instance_labels`）。

## SegmentationTask 说明
- 默认损失：`CrossEntropy(ignore_index=255)`（忽略无效像素）；也可配置 `dice` / `combined_segmentation`。
- 默认指标：mIoU + pixel_accuracy。
- 优化器 / 调度器通过配置 `training.optimizer` / `training.scheduler` 调整。

## 真实数据
把配置里的 `data.dataset` 换成 `cityscapes` / `ade20k` 并提供真实 mask 路径即可；
注意训练时同一 batch 内图像/掩码尺寸需一致（用分割变换统一 resize）。