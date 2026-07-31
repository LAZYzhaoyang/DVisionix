# 快速开始

> 所有命令请在已激活的 conda 环境 `dvisionix` 下执行。本机直接用 `python.exe` 可能报 `torch _C DLL load failed`，请使用 `conda activate dvisionix` 或 `conda run -n dvisionix ...`。

## 安装环境

```bash
conda create -n dvisionix python=3.14 -y
conda activate dvisionix
pip install torch torchvision timm numpy pillow opencv-python pyyaml tensorboard matplotlib onnx onnxruntime pytest
```

## 30 秒上手（合成数据，端到端）

```bash
conda run -n dvisionix python demos/train_from_config.py
```

该 demo 演示：**配置加载 → 数据集/模型/任务/回调 → 训练**，全部由一个 YAML 文件驱动，见
`configs/classification/demo_synthetic.yaml`。

## 手写训练三步

```python
from torch.utils.data import DataLoader
from dvisionix.data import build_dataset
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.models import SimpleCNN
from dvisionix.training import Trainer, ClassificationTask

train_ds = build_dataset({"type": "cifar10", "root": "./data", "train": True,
                                 transforms=ClassificationTransforms(train=True, image_size=32))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

model = SimpleCNN(num_classes=10)
task = ClassificationTask(num_classes=10, learning_rate=1e-3)
trainer = Trainer(task=task, train_loader=train_loader, device="auto", max_epochs=10)
trainer.fit(model)
```

## 相关文档
- [配置系统](config_system.md)
- [骨干网络（timm）](backbones.md)
- [日志系统](logging.md)
- [模型导出（ONNX）](model_export.md)
