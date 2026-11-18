# 快速开始

> 所有命令请在已激活的 conda 环境 `dvisionix` 下执行（`conda activate dvisionix` 或 `conda run -n dvisionix ...`）。

## 安装环境

```bash
conda create -n dvisionix python=3.10 -y
conda activate dvisionix
# 按你的 CUDA 版本安装 PyTorch，例如：
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 其余依赖（唯一来源是 pyproject.toml：[project] 核心 + 可选 extras）
pip install -e ".[full,dev]"
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

每个 trial 落在 `runs/search/trial_NNN/`（其下再按 `<experiment>/<时间戳>-<配置哈希>` 分层，
`history.csv` 在最内层），汇总结果写入 `runs/search/search_results.csv`
（含监控指标与 work_dir）。

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

### 性能开关与实验管理（v0.17-P3）
```yaml
training:
  compile: true        # torch.compile（DDP wrap 前编译；不可用时自动降级）
  channels_last: true  # 模型转 channels_last 内存格式（卷积网络友好）
  export_best_onnx: true  # 训练结束后把 checkpoints/best.pt 导出为 work_dir/best.onnx
```
- 新 run 目录自动带配置哈希后缀：`<timestamp>-<hash8>`（`hash_config`，同配置可复现定位）；
- 训练结束自动导出 `work_dir/best_metrics.csv`（按 checkpoint 监控指标选出的最优 epoch）。

## 相关文档
- [配置系统](config_system.md)
- [骨干网络](backbones.md)
- [日志系统](logging.md)
- [模型导出（ONNX）](model_export.md)
---

## v1.1 变更要点

> 完整清单见 [v1.1 变更与迁移指南](v1.1_changes.md)。

**安装方式**：依赖的唯一来源是 `pyproject.toml`（`setup.py` 已删除），
按需选择 extras：

```bash
pip install -e .[full,dev]     # 全部能力 + 开发工具
pip install -e .[export]       # 只要 ONNX 导出
```

**解释器**：`requires-python = ">=3.10"`，CI 覆盖 3.10 / 3.11 / 3.12。

**测试**：

```bash
pytest tests/                    # 全量（含官方配置端到端门禁，约 2 分钟）
pytest tests -m "not slow"       # 快速反馈
```

**新增性能基准工具**：

```bash
python tools/benchmark.py --list
python tools/benchmark.py --scenario all --json bench.json
```

**Windows 提示**：请先 `conda activate dvisionix`。直接用解释器绝对路径时
`Library\bin` 不在 `PATH`，子进程 `import torch` 会报 `DLL load failed`。

---

## v1.1.1 变更要点

> 完整清单见 [CHANGELOG](../CHANGELOG.md) 的 1.1.1 章节。

**`resume: true` 现在可用**（等价于 `auto`）。此前它会被当成 checkpoint 路径，
`torch.load(True)` 抛 `AttributeError: 'bool' object has no attribute 'seek'`，
而且工作目录解析还会另起一个带新时间戳的目录。详见
[配置系统](config_system.md) 的「v1.1.1 变更要点」。

**`--devices` 现在真正生效**：

```bash
# 2 卡：GPU0 与 GPU1
torchrun --nproc_per_node=2 tools/train.py --config configs/classification/demo_synthetic.yaml

# 指定用 1 号与 3 号卡（此前会被静默忽略，实际仍用 0、1 号卡）
torchrun --nproc_per_node=2 tools/train.py --config configs/classification/demo_synthetic.yaml \
  --devices 1,3
```

`torchrun` 下工作目录改用确定性的 `<配置哈希>` 命名（不带时间戳），保证所有 rank
落在同一目录；`history.csv` / `best_metrics.csv` / TensorBoard 只由 rank0 写出。

**超参搜索的汇总表现在有指标了**：此前 `search_results.csv` 的监控指标列**恒为空**
（`--work-dir` 是工作根，而汇总读的是 `work_dir/history.csv`，永远读不到）。

**断点续训在 GPU 上原本必然崩溃**（`map_location` 把 CPU 的 RNG 状态搬到了 GPU），
现已修复；`configs/classification/linear_eval.yaml` 在 GPU 上同理。

---

## 使用 DVX 标准数据集

自己的数据（分类 / labelme 检测与分割 / 多任务分类）建议先落成一份
`dvx_dataset.yaml`，再用 `DVXDataset` 加载。最短路径：

```bash
# 1. 扫描数据目录，生成规格草稿（只读，不移动/改名原始文件）
python tools/init_dataset.py --root ./data/my_cls --data-type classification \
  --out ./data/my_cls/dvx_dataset.yaml

# 2. 人工确认 classes 顺序（顺序即 label id）后体检，并预览切分
python tools/check_dataset.py --spec ./data/my_cls/dvx_dataset.yaml --split-preview

# 3. 加载
python -c "from dvisionix.data import DVXDataset; ds = DVXDataset('./data/my_cls/dvx_dataset.yaml', split='train'); print(len(ds), ds[0]['label'])"
```

- 字段含义、`ratio` / `dir` / `list` 三种切分、四类数据集输出结构见
  [DVX 标准数据集](dataset_spec.md)。
- 可直接复制的模板见 `configs/datasets/`：
  [分类](../configs/datasets/classification_folder.example.yaml)、
  [检测](../configs/datasets/detection_labelme.example.yaml)、
  [分割](../configs/datasets/segmentation_labelme.example.yaml)、
  [多任务](../configs/datasets/multi_task_classification.example.yaml)。
- 检测/分割用 `DataLoader` 时要传 `collate_fn=ds.collate_fn`；多任务分类用
  `task="<任务名>"` 过滤样本。
- 训练入口已支持 `data.dataset: dvx` + `data.spec`，见下一节；契约校验与自适应训练见
  [训练契约与自适应训练](training_contracts.md)。

---

## 用 DVX 标准数据集训练

`tools/train.py` 通过 `data.dataset: dvx` + `data.spec` 直接构建 `DVXDataset`，
`data.spec` 指向你的 `dvx_dataset.yaml`（四份模板里的 `data.spec` 是占位路径）：

```bash
# 分类
python tools/train.py \
  --config configs/examples/dvx_classification.yaml \
  --cfg-options data.spec=./data/my_cls/dvx_dataset.yaml \
  --work-dir /tmp/dvx_cls

# 检测
python tools/train.py \
  --config configs/examples/dvx_detection.yaml \
  --cfg-options data.spec=./data/my_det/dvx_dataset.yaml \
  --work-dir /tmp/dvx_det

# 分割
python tools/train.py \
  --config configs/examples/dvx_segmentation.yaml \
  --cfg-options data.spec=./data/my_seg/dvx_dataset.yaml \
  --work-dir /tmp/dvx_seg

# 路由式多任务分类（结构：<任务>/<类别>/*.jpg，data_type: multi_task_classification）
python tools/train.py \
  --config configs/examples/dvx_multi_task.yaml \
  --cfg-options data.spec=./data/my_multi/dvx_dataset.yaml \
  --work-dir /tmp/dvx_multi_task

# 多任务分类 2 卡 DDP（示例配置已写 training.find_unused_parameters: true，必须保留）
torchrun --nproc_per_node=2 tools/train.py \
  --config configs/examples/dvx_multi_task.yaml \
  --cfg-options data.spec=./data/my_multi/dvx_dataset.yaml \
  --work-dir /tmp/dvx_multi_task_ddp
```

- 单任务训练的 `model.num_classes` 必须与 spec 的类别数一致（分割通常含 `__background__`）；
  不一致会在训练开始前抛 `TaskDataMismatchError`，由契约校验兜底。多任务分类没有全局
  `model.num_classes`，类别数由 `model.heads` 自带并与 spec 的 `tasks` 逐任务比对。
- 多任务分类每个 batch 未必覆盖全部任务，DDP 必须开启
  `training.find_unused_parameters: true`（示例配置默认已开），否则会因未参与前向的分类头
  报 DDP 归约错误。完整说明（配置、指标命名、已知限制）见
  [路由式多任务分类](multi_task.md)。
- 数据变换由任务的 `build_transforms` 提供，检测/分割的 `collate_fn` 由入口自动使用。
- `training.device: auto` 按可用设备自动选择；AMP 仍由 `training.amp` 显式控制
  （示例配置均为 `false`）。
- 三级契约、8 个内置任务契约、自定义任务/模型声明与 `strategy: auto` 的行为见
  [训练契约与自适应训练](training_contracts.md)。

---

## 用训练好的模型推理

训练完成后，用 `tools/infer.py` 对单张图片或目录推理。完整 Trainer checkpoint
（`checkpoints/best.pt`）含非张量状态，需要 `--trusted`：

```bash
python tools/infer.py --config configs/classification/demo_synthetic.yaml \
  --checkpoint /path/to/work_dir/checkpoints/best.pt \
  --input ./samples --output ./outputs/cls --trusted --save-vis
```

每张图输出 `<stem>.json`，`--save-vis` 时另存 `<stem>_vis.jpg`；检测/分割把
`--config` 换成对应配置即可。Python 中可以直接构建预测器：

```python
from dvisionix.config import Config
from dvisionix.inference import ClassificationPredictor
from dvisionix.models import build_model

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
model = build_model(cfg.model.to_dict())
predictor = ClassificationPredictor.from_checkpoint(
    model, "/path/to/work_dir/checkpoints/best.pt", "classification",
    trusted=True, image_size=cfg.data.image_size,
)
result = predictor.predict("./samples/a.jpg")
print(result.topk, result.orig_size)
```

预处理规格、`Results` 字段、反变换与已知限制见 [推理](inference.md)。
