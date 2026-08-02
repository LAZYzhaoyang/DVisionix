# 骨干网络（Backbones）

`dvisionix.models.backbones` 提供 12 种即插即用的骨干网络，全部注册到 `BACKBONES` 注册表，
既可通过配置字典构建，也可直接 import 使用。骨干输出遵循统一契约：
**`features_only=True` 输出多尺度特征列表，否则输出全局池化特征向量 (B, C)**。

## 内置骨干清单

| 注册名 | 类 | 说明 |
|---|---|---|
| `sequential_backbone` | `SequentialBackbone` | 配置驱动的简单 CNN（教学/快速验证） |
| `convnext_backbone` | `ConvNeXtBackbone` | ConvNeXt（LN + 深度可分离 + LayerScale） |
| `convnextv2_backbone` | `ConvNeXtV2Backbone` | ConvNeXtV2（GRN 全局响应归一化） |
| `cspdarknet_backbone` | `CSPDarknetBackbone` | CSPDarknet（YOLOv5 风格） |
| `mobilenetv3_backbone` | `MobileNetV3Backbone` | MobileNetV3（MBConv + SE） |
| `efficientnet_lite_backbone` | `EfficientNetLiteBackbone` | EfficientNet-Lite（轻量 MBConv） |
| `vit_backbone` | `ViTBackbone` | ViT（patch embed + Transformer，正弦位置编码支持任意尺寸） |
| `swin_backbone` | `SwinBackbone` | Swin（window/shifted-window 注意力 + PatchMerging） |
| `swinv2_backbone` | `SwinV2Backbone` | SwinV2（cosine attention + 连续相对位置偏置 + res-post-norm） |
| `mit_backbone` | `MiTBackbone` | SegFormer 编码器（overlap patch embed + MixFFN） |
| `timm_backbone` | `TimmBackbone` | timm 数百种模型的特征提取封装 |
| `timm_classifier` | `TimmClassifier` | timm 模型 + 分类头（迁移学习） |

## 配置驱动使用

```yaml
# 以 ConvNeXt 为例（检测任务自动 features_only=True）
model:
  type: "fcos_detector"           # 或任意检测器/分割组合器
  backbone:
    type: "convnext_backbone"
    depths: [3, 3, 9, 3]
    dims: [96, 192, 384, 768]
```

## 编程式使用

```python
import torch
from dvisionix.models import ConvNeXtBackbone, TimmBackbone

# 多尺度特征（检测/分割）
bb = ConvNeXtBackbone(features_only=True)
feats = bb(torch.randn(2, 3, 224, 224))   # [stride 4/8/16/32 特征列表]

# 全局特征（分类）
bb2 = TimmBackbone("resnet50", pretrained=False)
feat = bb2(torch.randn(2, 3, 224, 224))   # (2, num_features)
```

## 预训练权重

- `timm_backbone` / `timm_classifier` 支持 `pretrained=True`（自动联网下载）。
- 组件化模型可通过 `model.pretrained_backbone` / `training.pretrained_backbone`
  加载任意 backbone 权重（含 Trainer 完整 checkpoint 与 EMA 导出，自动过滤 `backbone.` 前缀）。

## 迁移学习

```python
from dvisionix.models import TimmClassifier
model = TimmClassifier("resnet50", num_classes=10, pretrained=False)
model.freeze_backbone()    # 冻结骨干，仅训练分类头
# ... 训练后
model.unfreeze_backbone()  # 解冻做微调
```

## 列出 timm 可用模型

```python
from dvisionix.models import list_timm_models
print(list_timm_models("resnet*")[:5])
```

## 说明
- ViT/Swin 等 Transformer 骨干对输入分辨率有要求（如 224×224），请匹配 `image_size`。
- 所有骨干可自由与 necks（FPN / PANet / PixelDecoder）、heads（分类/检测/分割）组合，即插即用。
