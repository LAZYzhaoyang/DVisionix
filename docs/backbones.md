# 骨干网络（基于 timm）

`dvisionix.models` 提供对 [timm](https://github.com/huggingface/pytorch-image-models)
的封装，可直接使用 ResNet / ViT / Swin 等数百种模型。

## TimmClassifier（骨干 + 分类头）

```python
import torch
from dvisionix.models import TimmClassifier

model = TimmClassifier("resnet50", num_classes=10, pretrained=False)
logits = model(torch.randn(2, 3, 224, 224))   # (2, 10)

# 迁移学习：冻结骨干，仅训练分类头
model.freeze_backbone()
model.unfreeze_backbone()
```

## TimmBackbone（纯特征提取器）

```python
from dvisionix.models import TimmBackbone

backbone = TimmBackbone("swin_tiny_patch4_window7_224", pretrained=False)
feats = backbone(torch.randn(2, 3, 224, 224))  # (2, backbone.num_features)
print(backbone.num_features)
```

`TimmBackbone` 输出全局池化后的特征向量，可作为分类/检测/分割头的通用输入，实现即插即用。

## 列出可用模型

```python
from dvisionix.models import list_timm_models
print(list_timm_models("resnet*")[:5])
```

## 说明
- `pretrained=True` 会联网下载权重；离线/调试建议用 `pretrained=False`。
- ViT/Swin 等模型对输入分辨率有要求（如 224×224），请匹配 `image_size`。
