# 数据模块 (Data)

DVisionix 的数据层采用**统一接口 + 内部实现可替换**的设计：所有任务的数据集都遵循 `Sample` 协议，由 `BaseDataset` 统一驱动，原子变换（image / 几何同步 / 标签）可自由组合，主流公开数据集通过 `presets` 工具箱直接复用。

> 运行任何代码前请先 `conda activate dvisionix`。

## 设计总览

```
                ┌────────────────────────────────────────────────┐
                │                  build_dataset(cfg)            │
                │              (DATASETS registry factory)       │
                └────────────────────┬───────────────────────────┘
                                     │
       ┌────────────────────┬────────┴─────────┬────────────────────┐
       ▼                    ▼                  ▼                    ▼
   ImageFolder      CIFAR10 / 100         COCO / VOC /         Cityscapes /
   (分类)           (分类)                ADE20K (检测/分割)    ADE20K (分割)
       │                    │                  │                    │
       └────────────────────┴────────┬─────────┴────────────────────┘
                                     ▼
                          ┌────────────────────┐
                          │  BaseDataset       │  统一接口：
                          │  - samples         │  __getitem__(i) -> dict
                          │  - transforms      │  (内部按 task 注入 collate_fn)
                          │  - collate_fn      │
                          └─────────┬──────────┘
                                    ▼
                          ┌────────────────────┐
                          │  Sample 协议       │  必备字段：
                          │  (image/label/     │  - image  (H,W,3) uint8
                          │   boxes/labels/    │  - 任务字段 label / boxes+labels / mask
                          │   mask/meta)       │  - 可选 meta
                          └─────────┬──────────┘
                                    ▼
                          ┌────────────────────┐
                          │  TransformPipeline │  原子变换可自由组合
                          │  (image 原子 /     │  (image / box_sync / mask_sync)
                          │   box_sync /       │
                          │   mask_sync /      │
                          │   third-party)     │
                          └────────────────────┘
```

## Sample 协议

`Sample` 是 `dict` 的轻量子类。`sample.image` 与 `sample["image"]` 等价；transform 会按字段名缺失则**静默跳过**（如对没有 `mask` 字段的分类样本使用 `BoxSyncResize` 不会报错，`mask` 缺失则不缩放）。

| 字段      | 类型                           | 必含任务 | 说明                                   |
|-----------|--------------------------------|----------|----------------------------------------|
| `image`   | `np.ndarray` / `str` / `Tensor`| 所有     | 路径 / `uint8 (H,W,3)` / `(C,H,W)`    |
| `label`   | `int`                          | 分类     | 类别索引                               |
| `boxes`   | `np.ndarray float32 (N,4)`     | 检测     | xyxy 绝对坐标                          |
| `labels`  | `np.ndarray int64 (N,)`        | 检测     | 与 boxes 一一对应的类别                |
| `mask`    | `np.ndarray int64 (H,W)`       | 分割     | 单通道类别图                           |
| `meta`    | `dict`                         | 任意     | dataset 内部透传（原始尺寸、路径等）   |
| 其它      | 任意                           | 自定义   | 自定义任务可自由扩展                   |

## BaseDataset

```python
from dvisionix.data import BaseDataset, CustomDataset

samples = [
    {"image": "img_001.jpg", "label": 0},
    {"image": "img_002.jpg", "label": 1},
    ...
]
ds = CustomDataset(samples, transforms=my_pipeline, task_type="classification")
sample = ds[0]   # -> dict，已经过 transforms
```

关键约定：

- `samples` 是 `list[dict]`，每项至少含 `image` 字段（路径/numpy/tensor）。
- `transforms` 可以是 `TransformPipeline`、可调用对象或 `None`（仅做 `load_image`）。
- `collate_fn` 留 `None` 时由 `task_type` 自动注入（`detection` → `detection_collate`，`segmentation` → `segmentation_collate`，其它走 PyTorch 默认）。
- `load_image` 是 `sample -> np.ndarray` 的钩子，自定义读取逻辑（多帧、特殊格式等）只需覆盖它，不必继承重写。

### 继承 BaseDataset

需要扩展更多字段（如关键点、视频片段、深度图）时，直接继承 `BaseDataset`：

```python
from dvisionix.data import BaseDataset
from dvisionix.registry import DATASETS

@DATASETS.register()
class PoseDataset(BaseDataset):
    task_type = "keypoints"

    def __init__(self, root, transforms=None):
        samples = [...]  # 包含 image / keypoints / bbox 等字段
        super().__init__(samples, transforms=transforms, collate_fn=pose_collate)
```

也可以同时注册多个别名：

```python
@DATASETS.register()
@DATASETS.register(name="pose")
@DATASETS.register(name="keypoints")
class PoseDataset(BaseDataset): ...
```

## 原子变换 (Transforms)

所有变换继承 `BaseTransform`，实现 `__call__(Sample) -> Sample`，并通过 `@TRANSFORMS.register()` 注册到全局注册表。三大类：

| 模块              | 作用                                          | 主要类                                                                                 |
|-------------------|-----------------------------------------------|----------------------------------------------------------------------------------------|
| `image`           | 仅动 `image` 字段，任务无关                   | `ImageResize`, `RandomCrop`, `CenterCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`, `ImageNormalize`, `ToTensor` |
| `geometric`       | 同步更新 `image` / `boxes` / `mask`           | `BoxSyncResize`, `BoxSyncRandomHorizontalFlip`, `BoxSyncRandomCrop`, `BoxSyncPad`     |
| `labels`          | numpy -> torch.Tensor                          | `LabelToTensor`, `BoxesToTensor`, `MaskToTensor`                                       |
| `third_party`     | 适配 albumentations / kornia / torchvision     | `AlbumentationsWrapper`                                                                |
| 预设 pipeline     | 旧 API 兼容（同时支持 CamelCase 与 snake_case） | `ClassificationTransforms`, `DetectionTransforms`, `SegmentationTransforms`             |

### 自由组合

```python
from dvisionix.data import (
    ImageResize, RandomCrop, RandomHorizontalFlip, ColorJitter,
    BoxSyncResize, BoxSyncRandomHorizontalFlip, ImageNormalize, ToTensor,
    BoxesToTensor, TransformPipeline,
)

det_train = TransformPipeline([
    BoxSyncResize((640, 640)),
    BoxSyncRandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2),
    ImageNormalize(),
    ToTensor(),
    BoxesToTensor(),
])
```

### 配置 / 字符串构建

```python
from dvisionix.data import build_transform, build_pipeline

t = build_transform({"type": "image_resize", "size": [224, 224]})
p = build_pipeline([
    "random_hflip",                              # 仅写类名 -> 按无参构造
    {"type": "random_crop", "size": [224, 224]},
    "to_tensor",
])
```

## 公开数据集工具箱 (presets)

`dvisionix.data.presets` 内置了主流数据集的统一封装。**所有数据集都遵守相同的 `Sample` 协议**，可以直接喂给任意 `TransformPipeline`：

```python
from dvisionix.data import build_dataset, ClassificationTransforms

ds = build_dataset({
    "type": "cifar10",
    "root": "./data/cifar10",
    "train": True,
    "transforms": ClassificationTransforms(train=True, image_size=32),
})
```

可用数据集与 `type` 别名（任选其一）：

| 数据集           | type 名 (任选)                  | 适用任务        | 必需字段                                                  |
|------------------|---------------------------------|-----------------|-----------------------------------------------------------|
| ImageFolder      | `imagefolder` / `image_folder`  | 分类            | root，按类别分子目录                                      |
| CIFAR-10         | `cifar10`                       | 分类            | root，会自动下载                                          |
| CIFAR-100        | `cifar100`                      | 分类            | root，会自动下载                                          |
| ImageNet         | `imagenet`                      | 分类            | root/split，train/val 目录结构                            |
| COCO Detection   | `coco_detection` / `coco`       | 检测            | root + ann_file                                           |
| VOC Detection    | `voc_detection`                 | 检测            | root + (image_set / year)                                 |
| Cityscapes       | `cityscapes`                    | 分割            | root + split                                              |
| VOC Segmentation | `voc_segmentation`              | 分割            | root + (image_set / year)                                 |
| ADE20K           | `ade20k`                        | 分割            | root + split                                              |

> 真正的端到端训练仍推荐 `tools/train.py` + YAML 配置，详见 `docs/quick_start.md`。

## 自定义 transform

```python
from dvisionix.data import BaseTransform, Sample
from dvisionix.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register()
@TRANSFORMS.register(name="random_gray")
class RandomGray(BaseTransform):
    """以 p 概率把 image 灰度化（保持三通道）。"""

    name = "random_gray"

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            img = sample["image"]
            gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(img.dtype)
            sample["image"] = np.stack([gray] * 3, axis=-1)
        return sample
```

之后 `build_transform("random_gray")` 或 `@TRANSFORMS.register(name="random_gray")` 即可复用。

## 自定义 dataset

两种最常用姿势：

### 1. 用 CustomDataset (samples + transforms 即可)

只要 `samples` 是 `list[dict]`，按任务填字段就行：

```python
from dvisionix.data import CustomDataset, ClassificationTransforms

samples = [
    {"image": "a.jpg", "label": 0},
    {"image": "b.jpg", "label": 1},
    ...
]
ds = CustomDataset(samples, task_type="classification",
                   transforms=ClassificationTransforms(train=True))
```

### 2. 继承 BaseDataset (需要自定义加载/字段)

```python
import numpy as np
from dvisionix.data import BaseDataset
from dvisionix.registry import DATASETS

@DATASETS.register()
class DepthEstimationDataset(BaseDataset):
    task_type = "depth"

    def __init__(self, root, split="train", transforms=None):
        # 自定义 samples 组织：每项包含 image / depth (路径) / 其它元信息
        samples = self._scan(root, split)
        super().__init__(samples, transforms=transforms, return_meta=True)

    def _scan(self, root, split):
        ...

    def load_image(self, sample):
        # 读取 RGB 图
        import cv2
        bgr = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth(self, sample):
        # 读取 depth (H, W) float32（自定义字段不在 BaseDataset 处理范围）
        import cv2
        return cv2.imread(sample["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32)

    def __getitem__(self, idx):
        raw = dict(self.samples[idx])
        raw["image"] = self.load_image(raw)
        if "depth" in raw:
            raw["depth"] = self.load_depth(raw)
        if self.transforms is not None:
            raw = self.transforms(raw)
        return raw
```

> **最小可工作单元**：在自定义任务下，你至少需要给数据集（`BaseDataset` 子类或 `CustomDataset`）提供：(1) `samples` 列表（每项是 `dict`、含 `image` 字段），(2) 任务字段（`label` / `boxes`+`labels` / `mask` / 自定义），(3) `transforms`（原子变换按任意顺序拼装），(4) `collate_fn`（变长 pad 时需要，例如 `detection_collate`）。其它（`load_image`、字段扩展、collator、注册别名）都属于"按需增强"。

## 自定义任务的数据集 (典型流程)

把"分类 / 检测 / 分割"之外的任意任务（例如关键点检测、深度估计、跨模态检索）接到 `DVisionix` 的训练循环，需要做 4 件事：

1. **定义 `Sample` 字段**：在 `samples` 字典里塞进你的任务字段（`keypoints` / `depth` / `caption` 等），与 `image` 同级存放即可。`Sample` 协议不限制新字段。

2. **写一个 `BaseDataset` 子类**（或用 `CustomDataset`）：

   - 构造 `samples` 列表；
   - 必要时覆盖 `load_image` 或新增 `load_<your_field>`；
   - 设置 `task_type` 和 `collate_fn`。

3. **组装 transforms**：从原子变换池里挑出处理 `image` 的部分，再加你自己的自定义变换（处理 `keypoints` / `depth` 等）。所有变换都要实现 `__call__(Sample) -> Sample` 并通过 `@TRANSFORMS.register()` 注册。

4. **注册到 `DATASETS`**：用 `@DATASETS.register()` + 自定义 `name`，之后 `build_dataset({"type": "your_dataset", ...})` 即可像内置数据集一样使用。

```python
# 自定义任务最小骨架
import torch
from dvisionix.data import BaseDataset, BaseTransform, ImageResize, ImageNormalize, ToTensor, TransformPipeline
from dvisionix.registry import DATASETS, TRANSFORMS

@TRANSFORMS.register()
class DepthToTensor(BaseTransform):
    name = "depth_to_tensor"
    def __call__(self, sample):
        if "depth" in sample and not isinstance(sample["depth"], torch.Tensor):
            sample["depth"] = torch.as_tensor(sample["depth"], dtype=torch.float32)
        return sample

@DATASETS.register()
class DepthEstimationDataset(BaseDataset):
    task_type = "depth"
    def __init__(self, root, transforms=None):
        samples = [...]  # 每个 dict 含 image + depth
        super().__init__(samples, transforms=transforms, return_meta=True,
                         collate_fn=depth_collate)

pipeline = TransformPipeline([
    ImageResize((256, 256)),
    ImageNormalize(),
    ToTensor(),
    DepthToTensor(),
])
ds = DepthEstimationDataset("./data/depth", transforms=pipeline)
```

## 验证

- `pytest tests/test_data/` - 数据模块单元测试
- `pytest tests/` - 全量测试
---

## v1.1 变更要点

> 完整清单见 [v1.1 变更与迁移指南](v1.1_changes.md)。

**裁剪契约**：尺寸不足时默认**直接报错**，不再静默返回错误尺寸。

```python
from dvisionix.data.transforms import RandomCrop, CenterCrop, BoxSyncRandomCrop

RandomCrop((224, 224))                     # 输入 < 224 时抛 ValueError
RandomCrop((224, 224), on_small="pad")     # 右下补零
CenterCrop((224, 224), on_small="resize")  # 直接缩放
BoxSyncRandomCrop((640, 640))              # 几何版只支持 "error" / "pad"
```

内置预设管线不受影响：分类与检测的训练管线现在都**先 resize 到 1.1× 再随机裁剪**
（v1.0.0 的检测管线是「resize 到目标尺寸 → 同尺寸裁剪」，随机偏移恒为 0，
增强完全没有生效）。

**mask 一定是 long**：`ToTensor` 对 mask 与维度无关地输出 `torch.long`；
`MaskToTensor` 无论输入是 numpy 还是 Tensor 都强制转 long，并会拒绝
非整值浮点、负值与非法形状 —— 不再把错误推迟到 `CrossEntropyLoss`。

**几何变换入口校验 boxes 与 labels 数量一致**，不一致直接报错，
不会产出标签与框错位的样本。

**合成数据**：`tools/train.py` 的 train / val 使用不同 split 前缀与独立随机种子
（v1.0.0 中 val 就是 train 的前 N 张，指标含数据泄漏），且现在**可复现**。

**Sample 字段名拼写检查**：`Sample.unknown_keys()` 返回契约外的字段名，
`BaseDataset` 会为每个未知字段名告警一次（每个字段只报一次），
用于捕捉 `bboxes` 这类会静默失效的拼写错误。自定义字段请加入
`Sample.EXTENDED_KEYS`。

**禁止流水线内重复归一化**：`TransformPipeline` 会拒绝包含两个
`provides_normalization=True` 变换的组合（此前该标记被聚合但从未被消费）。
