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

## 训练工程增强（v0.16.0）
- 调度器 warmup：`training.scheduler: {type: linear_warmup, warmup_epochs: 3, scheduler: {type: cosine, T_max: 100}}`
- 梯度裁剪：`training.gradient_clip_value`（值裁剪）或 `training.gradient_clip_val`（范数裁剪）
- EMA：`training.callbacks` 配置 `{type: ema, decay: 0.999, decay_warmup_epochs: 5, save_final: true}`（结束后导出 `work_dir/ema_last.pt`）
- 线性评估（自监督表征质量）：`model.pretrained_backbone` 加载预训练权重后冻结 encoder + L2 归一化线性头
  （见 `configs/classification/linear_eval.yaml`）

## 训练工程增强（v0.17.0）

### 超参搜索（tools/hparam_search.py）
参数网格用 YAML 描述（点路径 -> 候选值），逐 trial 独立进程跑 `tools/train.py`，实验完全隔离：

```yaml
# configs/classification/hparam_search.yaml
training.optimizer.lr: [0.0001, 0.001, 0.003]
training.optimizer.weight_decay: [0.0001, 0.001]
training.scheduler.type: [cosine, step]
```

```bash
# 笛卡尔积全组合
python tools/hparam_search.py --config configs/classification/demo_synthetic.yaml \
  --param-spec configs/classification/hparam_search.yaml --monitor val_accuracy --mode max

# 随机采样 8 组
python tools/hparam_search.py --config configs/classification/demo_synthetic.yaml \
  --param-spec configs/classification/hparam_search.yaml --num-trials 8 --seed 0
```

每个 trial 落在 `runs/search/trial_NNN/`，汇总结果写入 `runs/search/search_results.csv`（含监控指标与 work_dir）。

### 特征蒸馏（FeatureDistillLoss）
```python
from dvisionix.models.losses import FeatureDistillLoss
from dvisionix.training import DistillCallback

# teacher 中间特征（如 encoder 输出列表）挂在 trainer 上：
cb = DistillCallback(
    teacher=teacher_model,
    temperature=4.0,
    feature_extractor=lambda m, x: [m.encoder(x)],
)
# 任务内：loss = FeatureDistillLoss(normalize=True)(student_feats, trainer.teacher_features)
```

## 相关文档
- [配置系统](config_system.md)
- [骨干网络（timm）](backbones.md)
- [日志系统](logging.md)
- [模型导出（ONNX）](model_export.md)