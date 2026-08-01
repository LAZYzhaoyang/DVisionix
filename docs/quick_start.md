# 快速开始

> 所有命令请在已激活的 conda 环境 `dvisionix` 下执行（`conda activate dvisionix` 或 `conda run -n dvisionix ...`）。

## 安装环境

```bash
conda create -n dvisionix python=3.10 -y
conda activate dvisionix
# 按你的 CUDA 版本安装 PyTorch，例如：
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 其余依赖
pip install opencv-python numpy pyyaml tensorboard matplotlib timm onnx onnxruntime pytest
```

## 30 秒上手：Config 驱动训练（合成数据）

```bash
conda run -n dvisionix python tools/train.py --config configs/classification/demo_synthetic.yaml
```

一条命令完成：**配置加载 → 数据集/模型/任务/回调/日志 → 训练**。
训练产物（日志 / TensorBoard / 检查点 / 最终配置）全部落在工作目录
`~/dvisionix_runs/<experiment>/<时间戳>/`（可通过 `--work-dir` 或环境变量 `DVISIONIX_WORK_DIR` 覆盖）。

常用选项：

```bash
# 覆盖单个参数
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --cfg-options training.num_epochs=5 training.optimizer.lr=0.01

# 自动续训最近一次任务（或 --resume <checkpoint 路径>）
python tools/train.py --config configs/classification/demo_synthetic.yaml --resume auto

# 多卡训练（需 2+ 张 GPU）
torchrun --nproc_per_node=2 tools/train.py --config configs/classification/demo_synthetic.yaml --devices 0,1
```

## 编程式训练三步

```python
from torch.utils.data import DataLoader
from dvisionix.data import build_dataset
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.models import SimpleCNN
from dvisionix.training import Trainer, ClassificationTask

train_ds = build_dataset(
    {"type": "cifar10", "root": "./data", "train": True,
     "transforms": ClassificationTransforms(train=True, image_size=32)}
)
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