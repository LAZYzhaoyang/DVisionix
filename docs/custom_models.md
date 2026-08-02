# 自定义 Layer 与 Model

DVisionix 的模型层采用组件化 + 注册表的设计：`layers`（层）→ `backbones`（骨干）→ `necks`（颈部）→ `heads`（头部）→ `models`（整体模型）。所有组件都注册到全局注册表，既可直接 import 使用，也可通过配置字典按名称构建。

本文说明如何：
1. 使用/自定义 layer，并封装 timm 的 layer；
2. 使用/自定义 model；
3. 通过配置驱动组装模型。

---

## 一、Layer 模块

### 1.1 内置层

`dvisionix.models` 直接导出以下内置层（均已注册到 `LAYERS`）：

| 层 | 注册名 | 说明 |
| --- | --- | --- |
| `ConvNormAct` | `conv_norm_act` | Conv2d + Norm + Activation 组合块 |
| `MLP` | `mlp` | 两层前馈网络（FFN） |
| `SEBlock` | `se` | Squeeze-and-Excitation 通道注意力 |
| `DropPath` | `drop_path` | 随机深度（Stochastic Depth） |

```python
import torch
from dvisionix.models import ConvNormAct, SEBlock

block = ConvNormAct(3, 16, kernel_size=3, stride=2, norm="bn", act="relu")
x = torch.randn(2, 3, 32, 32)
y = block(x)            # (2, 16, 16, 16)
y = SEBlock(16)(y)      # 通道注意力，形状不变
```

### 1.2 norm / 激活的按名构建

避免到处写 `if name == ...`，用 `build_norm_layer` / `build_activation_layer` 按字符串或配置字典构建：

```python
from dvisionix.models import build_norm_layer, build_activation_layer

norm = build_norm_layer("bn", 64)                              # BatchNorm2d(64)
norm = build_norm_layer({"type": "gn", "num_groups": 32}, 64)  # GroupNorm
act = build_activation_layer("silu")                           # SiLU
```

- norm 支持：`bn` / `bn1d` / `gn` / `in` / `ln`（`None` 返回 `Identity`）。
- act 支持：`relu` / `relu6` / `leaky_relu` / `elu` / `gelu` / `silu` / `sigmoid` / `tanh` / `hardswish` / `identity`（`None` 返回 `Identity`）。
- GroupNorm 的 `num_groups` 会自动调整为能整除通道数的值。

### 1.3 配置驱动构建 layer

```python
from dvisionix.models import build_layer

layer = build_layer({"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2})
layer = build_layer({"type": "se", "channels": 64, "reduction": 8})
```

`build_layer` 从全局 `LAYERS` 注册表按 `type` 查找并实例化，其余字段作为构造参数传入。

### 1.4 封装 timm 的 layer

timm 提供大量高质量层（`SqueezeExcite` / `DropPath` / `Mlp` / `ConvNormAct` 等）。DVisionix 用统一接口暴露：

```python
from dvisionix.models import create_timm_layer, list_timm_layers

# 按名称实例化 timm.layers 中的任意层或工厂函数
se = create_timm_layer("SqueezeExcite", 64, rd_ratio=0.25)
dp = create_timm_layer("DropPath", drop_prob=0.1)
conv = create_timm_layer("create_conv2d", 32, 64, kernel_size=3)

# 查看可用层
print(list_timm_layers()[:10])
```

部分常用 timm 层还以 `timm_` 前缀注册到 `LAYERS`，可直接用配置构建（避免与内置同名层冲突）：

```python
from dvisionix.models import build_layer

se = build_layer({"type": "timm_squeeze_excite", "channels": 64})
dp = build_layer({"type": "timm_drop_path", "drop_prob": 0.1})
```

> 注意：timm 为可选依赖，仅在实际调用 timm 相关接口时才要求安装（`pip install timm`）。

### 1.5 自定义 layer 并注册

仿照 `dvisionix/models/layers/basic.py`，定义 `nn.Module` 并用装饰器注册即可：

```python
import torch.nn as nn
from dvisionix.registry import LAYERS
from dvisionix.models import build_norm_layer, build_activation_layer

@LAYERS.register()                 # 以类名 "ResidualBlock" 注册
@LAYERS.register(name="resblock")  # 再注册一个小写别名
class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="bn", act="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = build_norm_layer(norm, channels)
        self.act = build_activation_layer(act)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = build_norm_layer(norm, channels)

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + x)

# 注册后即可用配置构建
from dvisionix.models import build_layer
block = build_layer({"type": "resblock", "channels": 64})
```

要让自定义层在包导入时自动注册，把定义放进被导入的模块（例如新建 `dvisionix/models/layers/my_layers.py` 并在 `layers/__init__.py` 里 import）。

---

## 二、Model 模块

### 2.1 模型契约（BaseModel）

所有模型继承 `dvisionix.models.BaseModel`：

- `forward` 只返回**原始预测**（logits / raw 张量），不做 NMS / decode 等后处理；
- `task_type` 取 `classification` / `detection` / `segmentation` 之一；
- 提供 `count_parameters` / `freeze` / `unfreeze` / `get_device` 等通用能力；
- 可选实现 `init_weights`（权重初始化）与 `from_config`（配置化构建）。

### 2.2 自定义 model 并注册

```python
import torch.nn as nn
from dvisionix.models import BaseModel, ConvNormAct
from dvisionix.registry import MODELS

@MODELS.register()
@MODELS.register(name="my_cnn")
class MyCNN(BaseModel):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.stem = ConvNormAct(in_channels, 32, stride=2)
        self.block = ConvNormAct(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

# 用配置构建
from dvisionix.models import build_model
model = build_model({"type": "my_cnn", "num_classes": 10})
```

### 2.3 组件化组装（backbone + neck + head）

复杂模型推荐用组件化模型把 backbone / neck / head 组合起来（分类 `linear_classifier`、分割 `segmentation_model`、检测 `fcos` / `retinanet` / `yolo` / `detr` / `rtdetr`），各组件都可用配置指定：

```python
from dvisionix.models import build_model

model = build_model({
    "type": "segmentation_model",
    "backbone": {"type": "TimmBackbone", "name": "resnet18",
                 "features_only": True, "out_indices": [4]},
    "head": {"type": "DeepLabV3Head", "num_classes": 19},
})
```

### 2.4 组件接口契约（backbone / neck / head）

组件化模型（`SingleStageDetector` / `SegmentationModel` / `LinearClassifier`）按 `backbone -> neck -> head` 组装组件，并在组件间自动传递通道数。自定义这些组件时，除了继承 `BaseModel`、实现 `forward` 外，还需暴露以下属性/接口：

| 组件 | 构造参数（至少） | 必须暴露的属性 | `forward` 输入 → 输出 |
| --- | --- | --- | --- |
| backbone | 由配置提供 | `out_channels: List[int]`、`num_features: int` | 图像 `(B, C, H, W)` → 分类模式返回 `(B, num_features)`；`features_only` 模式返回 `List[Tensor]`（多尺度特征图） |
| neck | `in_channels: List[int]`（由 backbone 注入） | `out_channels`（int 或 `List[int]`） | `List[Tensor]` → `List[Tensor]` |
| head | `in_channels`（由上游注入）、`num_classes` | — | 分类：`(B, in_channels)` → `(B, num_classes)`；分割/检测：`(B, C, H, W)` → 原始预测图 |

通道自动传递规则（见 `dvisionix/models/detectors/base.py`）：
- `backbone.out_channels` → `neck.in_channels`；
- 有 neck 时 `neck.out_channels`（取最后一层）→ `head.in_channels`；无 neck 时分类用 `backbone.num_features`、检测/分割用 `backbone.out_channels[-1]` → `head.in_channels`；
- 非分类任务会自动给 backbone 设置 `features_only=True`（可在配置中显式覆盖）。

> 提示：`in_channels` 由组件化模型自动注入到 head 配置中，因此 head 的构造函数必须接受 `in_channels` 参数；
> 多尺度头（`UNetDecoder` / `SegFormerHead` / `MaskFormerHead` / `RTDETRHead`）例外：注入 `in_channels_list`。

> 自定义检测器 / 分割头时，专属 decode（预测 → boxes/scores/labels 或 masks）建议写在模型/head 同文件中
> （如 `detectors/fcos.py` 的 `fcos_decode`），`postprocess.py` 只保留共享原语 `nms / batched_nms / box_iou`。

### 2.5 用 layers 拼装自定义 backbone（SequentialBackbone）

如果只是想把若干 layer 顺序堆叠成一个骨干网络，不必手写类——用内置的 `SequentialBackbone` 按配置列表拼装即可，它会自动 dry-run 推导并暴露 `out_channels` / `num_features`：

```python
import torch
from dvisionix.models import SequentialBackbone

stages = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
    # 一个 stage 也可以是配置列表，会被组合成一个 nn.Sequential
    [
        {"type": "conv_norm_act", "in_channels": 64, "out_channels": 128, "stride": 2},
        {"type": "se", "channels": 128},
    ],
]

# 多尺度特征图（供 FPN / 检测 / 分割）
backbone = SequentialBackbone(stages, features_only=True)
feats = backbone(torch.randn(2, 3, 64, 64))   # List[Tensor]
print(backbone.out_channels)                  # [32, 64, 128]

# 全局特征向量（供分类头）
backbone = SequentialBackbone(stages, features_only=False)
vec = backbone(torch.randn(2, 3, 64, 64))     # (2, 128)
```

配合 `LinearClassifier` 纯配置组装分类模型：

```python
from dvisionix.models import build_model

model = build_model({
    "type": "linear_classifier",
    "backbone": {"type": "sequential_backbone", "stages": stages},
    "num_classes": 10,
})
out = model(torch.randn(2, 3, 64, 64))        # (2, 10)
```

- `stages`: 每个元素是一个 layer 配置字典，或配置字典列表（列表组合为一个 stage）。
- `features_only`: `True` 返回多尺度特征图列表并暴露 `out_channels`；`False` 返回全局池化向量。
- `out_indices`: `features_only` 模式下选择返回哪些 stage（默认全部）。

### 2.4 在 YAML 配置中使用

`tools/train.py` 通过 `model` 段构建模型，可直接引用注册名与嵌套组件：

```yaml
model:
  type: linear_classifier
  num_classes: 10
  backbone:
    type: TimmBackbone
    name: resnet18
    features_only: false
  head:
    type: ClsHead
    num_classes: 10
```

自定义的 layer / model 只要在训练脚本导入前完成注册（例如放在被 import 的模块里），即可在配置中按 `type` 名引用。

---

## 三、注册表一览

| 注册表 | 构建函数 | 用途 |
| --- | --- | --- |
| `LAYERS` | `build_layer` | 层（ConvNormAct / SE / timm 层等） |
| `BACKBONES` | — | 骨干网络（TimmBackbone / SequentialBackbone） |
| `NECKS` | `build_neck` | 颈部（FPN） |
| `HEADS` | `build_head` | 头部（Cls/Seg/Det Head） |
| `MODELS` | `build_model` | 整体模型 |

所有注册表都在 `dvisionix.registry`，可用 `list(LAYERS.keys())` 查看已注册名称。