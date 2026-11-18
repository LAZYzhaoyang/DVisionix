# 路由式多任务分类

> 适用范围：v2.0.0（P3 已完成）。本页面向使用者，说明如何用
> `MultiTaskClassifier` + `MultiTaskClassificationTask` 在
> `data_type: multi_task_classification` 的 DVX 数据集上训练。
>
> 接口定义以实现契约 [P3 实现契约](design/v2_p3_contract.md) 为准，本页只描述实际行为。
> 相关文档：[DVX 标准数据集](dataset_spec.md)、[训练契约与自适应训练](training_contracts.md)、
> [快速开始](quick_start.md)、[指标](metrics.md)、[配置系统](config_system.md)。

---

## 1. 概念：共享骨干 + 按样本路由

多任务分类解决的是「**同一份数据里包含多个彼此独立的分类任务**」的场景，例如车牌数据集
按任务分成 `licence`（无 / 短牌照 / 长牌照）与 `logo`（一体奇瑞标 / 嵌入奇瑞标）两个任务
目录：每个样本只属于一个任务，但不同任务的标签空间互不影响。

它的结构是：

```text
                     ┌─ task == "licence" → licence head → (n_licence, 3)
image ── backbone ──►│      （同一份特征）
        （只前向一次） └─ task == "logo"    → logo head    → (n_logo, 2)
```

三个关键点：

1. **共享骨干只前向一次**。`MultiTaskClassifier.forward` 先对整批图片跑一次
   `backbone`，再按每个样本的 `task` 用 `index_select` 取特征子集，送进对应分类头。
   多任务不会把骨干计算量乘以任务数，也不会为每个任务重复过骨干。
2. **按样本路由**。batch 里每个样本都带自己的任务名（`batch["task"]`，`list[str]`），
   不是「整个 batch 属于同一个任务」。同一个 batch 可以混有不同任务的样本。
3. **各任务标签空间独立**。每个 head 自己声明 `num_classes`，`licence` 的 `0/1/2`
   与 `logo` 的 `0/1` 互不影响；模型没有全局 `model.num_classes`。

它与相近概念的区别：

| 场景 | 样本标签 | 本页方案 |
|---|---|---|
| 单任务分类 | 一张图一个 `label`，整个数据集一张类表 | `ClassificationTask` |
| 多标签分类 | 一张图对应多个**同一类表内**的标签 | `MultiLabelTask` |
| **路由式多任务**（本页） | 一张图属于**一个**任务，但任务间类表彼此独立 | `MultiTaskClassificationTask` |
| 检索 / re-ID | 需要 embedding 与检索评估 | 未实现 |

> P3 的范围**只有训练**。多任务模型的推理、导出专项（ONNX 等）与跨任务共享标签空间都未实现，
> 详见 §7。

---

## 2. 数据要求

### 2.1 规格与目录结构

训练数据必须是一份 DVX 规格，且：

```yaml
data_type: multi_task_classification
layout: folder_per_class
```

标准目录结构是 **`<任务>/<类别>/*.jpg`**：每个任务一个任务目录，任务目录下每个类别一个
类别目录，图片放在类别目录里（类别目录下还可以再嵌套）：

```text
my_multi/                        # 数据集根目录（root）
├── dvx_dataset.yaml
├── licence/                     # <任务>/，对应 tasks[*].path
│   ├── 无/                      # <类别>/，对应 tasks[*].classes
│   ├── 短牌照/
│   └── 长牌照/
└── logo/
    ├── 一体奇瑞标/
    └── 嵌入奇瑞标/
```

对应的规格（完整字段见 [DVX 标准数据集](dataset_spec.md) 的 §3.2 与 §6.7）：

```yaml
version: 1
data_type: multi_task_classification
layout: folder_per_class
root: .

tasks:
  - name: licence
    path: licence
    classes: [无, 短牌照, 长牌照]      # 顺序即该任务的 label id
  - name: logo
    path: logo
    classes: [一体奇瑞标, 嵌入奇瑞标]

splits:
  mode: ratio
  ratio: {train: 0.8, val: 0.2}
  seed: 42
  stratify: class                     # 多任务按 (task, label) 分层
```

要点：

- **每个任务的 `classes` 独立编号**，都从 0 开始；样本同时带 `task` 名与在该任务内的 `label`。
- 多任务规格**不允许**顶层 `classes`；非多任务规格也不允许写 `tasks`。
- 不同任务的 `path` 不能重叠：同一张图片只能属于一个任务，否则扫描报
  `AnnotationError`（重复 `rel_path`）。
- 标准结构不需要额外的 `split_dirs/` 中间层；`ratio` 切分在内存完成，按 `(task, label)`
  分层，不移动任何文件。若数据本身已按目录切好，也可用 `dir` 模式，目录相对每个 `task.path` 解析。
- 数据体检、三种切分与全部字段说明见 [DVX 标准数据集](dataset_spec.md)；
  可复制的模板见
  [multi_task_classification.example.yaml](../configs/datasets/multi_task_classification.example.yaml)。

### 2.2 先体检再训练

```bash
# 生成草稿（--task-dir 可重复；不写时自动发现一级任务目录）
python tools/init_dataset.py --root ./data/my_multi \
  --data-type multi_task_classification \
  --task-dir licence --task-dir logo \
  --out ./data/my_multi/dvx_dataset.yaml

# 人工确认每个任务的 classes 顺序后体检，并预览切分
python tools/check_dataset.py --spec ./data/my_multi/dvx_dataset.yaml --split-preview
```

若把多个任务目录写错（例如 `logo` 的 `path` 指向了 `licence` 的子树），
扫描阶段就会报重复 `rel_path`，不要靠训练去发现这类问题。

---

## 3. 完整配置示例

仓库中的模板是
[dvx_multi_task.yaml](../configs/examples/dvx_multi_task.yaml)，内容如下
（`data.spec` 是占位路径，使用前改成你的规格）：

```yaml
experiment_name: "dvx_multi_task"

# 训练任务类型（权威来源：dvisionix/config/schema.py::TASK_TYPES）
task_type: "multi_task_classification"

model:
  type: "multi_task_classifier"
  # 共享骨干：只做一次 backbone 前向，各 head 在特征上按样本路由
  backbone:
    type: "mobilenetv3_backbone"
    features_only: false
  # 每个任务一个分类头；num_classes = 该任务的类别数
  heads:
    licence:
      type: "cls_head"
      num_classes: 3
    logo:
      type: "cls_head"
      num_classes: 2

task:
  type: "MultiTaskClassificationTask"

data:
  dataset: "dvx"
  # 占位路径：指向你的多任务 dvx_dataset.yaml（root 相对 spec 所在目录解析）
  spec: "./data/multicls/dvx_dataset.yaml"
  image_size: 224

training:
  num_epochs: 1
  batch_size: 16
  amp: false
  num_workers: 0
  early_stopping: {enabled: false}
  # 多卡必须开启，原因见 §4.2
  find_unused_parameters: true

# 损失 / 指标由任务按任务名内部创建（每任务一个 CrossEntropy / MetricCollection）
loss: null
metrics: null
```

逐项说明：

| 字段 | 说明 |
|---|---|
| `experiment_name` | 实验名，用于工作目录与日志。 |
| `task_type` | 必须是 `multi_task_classification`；它是训练任务类型，不是模型任务能力。 |
| `model.type` | `multi_task_classifier`（类名 `MultiTaskClassifier`，两种注册名都可用）。 |
| `model.backbone` | 走 `BACKBONES.build`；未写 `features_only` 时模型会注入 `false`。特征维度取骨干的 `num_features`（兜底 `out_channels[-1]`）。 |
| `model.heads` | `{任务名: 头配置}`，任务名即 `batch["task"]` 的取值与指标前缀。每个头**必须**写 `num_classes`；`type` 缺省为 `cls_head`，`in_channels` 自动注入。 |
| `model.num_classes` | **不要写**。多任务没有全局类别数；模型只读属性 `num_classes` 是 `{任务: 类别数}` 的 dict，由 heads 自动产生。 |
| `task.type` | `MultiTaskClassificationTask`（注册名即类名）。 |
| `task.task_loss_weights` | 可选，`{任务名: 权重}`，用于加权求和；未列出的任务权重为 1.0。见 §5.3。 |
| `task.num_classes` | 可选，`{任务名: 类别数}`；只在模型没有 `num_classes` 字典时给「本 batch 未出现任务」的空占位定宽。正常用 `MultiTaskClassifier` 时不需要写。 |
| `data.dataset` | 固定 `dvx`；训练入口用 `data.spec` 构建 `DVXDataset`。 |
| `data.spec` | 多任务 `dvx_dataset.yaml` 路径；相对路径按**当前工作目录**解析，规格里的 `root` 相对规格文件所在目录解析。 |
| `data.image_size` | 图像尺寸；变换由任务的 `build_transforms` 提供（复用 `ClassificationTransforms`）。 |
| `training.num_epochs` / `batch_size` / `amp` / `num_workers` | 常规训练参数；示例配置为 1 epoch、bs 16、关闭 AMP。 |
| `training.find_unused_parameters` | **多卡必须 `true`**（见 §4.2）；单卡下不生效。 |
| `loss: null` / `metrics: null` | 不向任务注入全局损失/指标配置：任务内部默认**共享一个** `CrossEntropy`，并为每个任务维护独立的分类指标集合。 |

> 注：模板里的注释「每任务一个 CrossEntropy / MetricCollection」描述的是**逐任务计算损失、
> 逐任务维护指标**；默认实现是所有任务共享同一个 `CrossEntropy` 实例（可用顶层 `loss` 替换），
> 指标集合才是每个任务各自一份。

可选的进一步定制：

```yaml
task:
  type: "MultiTaskClassificationTask"
  task_loss_weights: {licence: 1.0, logo: 0.5}   # 总损失按权重求和

# 逐任务指定指标模板；也可写成 list / 单个模板让所有任务共用
metrics:
  licence: {type: accuracy}
  logo: {type: f1, average: macro}
```

指标配置的完整语法见 [指标](metrics.md)；`loss` 的写法见
[训练契约与自适应训练](training_contracts.md) 的配置部分。

### 3.1 与 spec 的一致性

`model.heads` 的键集合与每个头的 `num_classes` 必须与 spec 的 `tasks` 完全一致：

- `heads` 缺少 spec 里的任务、或多出 spec 没有的任务；
- 某个任务的 `num_classes` 不等于该任务 `classes` 的长度；

都会在训练开始前由契约校验拦下（`TaskDataMismatchError`，消息列出具体差异），
不会等到 loss 形状报错。见 §6。

---

## 4. 训练

以下命令在仓库根目录执行，使用已激活的 `dvisionix` 环境
（本机验证用 `/data/lzy/Env/condaEnvs/dvisionix/bin/python`）：

### 4.1 单卡

```bash
python tools/train.py \
  --config configs/examples/dvx_multi_task.yaml \
  --cfg-options data.spec=./data/multicls/dvx_dataset.yaml \
  --work-dir /tmp/dvx_multi_task --force
```

- `--force` 跳过「自动续训最近一次任务」的探测，确保从零开始；需要续训时去掉它并加 `--resume auto`。
- 工作目录落到 `--work-dir` 下的 `<experiment>/<时间戳>-<配置哈希>/`，`history.csv` /
  `best_metrics.csv` / 检查点 / 日志都在其中。
- 日志会先打印 `Train batches: N, Val batches: M`，随后每个 epoch 打印带任务前缀的汇总。

### 4.2 2 卡 torchrun（必须开 `find_unused_parameters`）

```bash
torchrun --nproc_per_node=2 tools/train.py \
  --config configs/examples/dvx_multi_task.yaml \
  --cfg-options data.spec=./data/multicls/dvx_dataset.yaml \
  --work-dir /tmp/dvx_multi_task_ddp --force
```

`strategy` 默认 `auto`，torchrun 下会自动识别为 `ddp`；需要指定卡时加 `--devices 0,1`。

**为什么必须 `training.find_unused_parameters: true`**：

- 每个样本只属于一个任务，而 DataLoader 是随机打乱的，因此**每个 batch 未必覆盖全部任务**
  （可能只有 `licence`、可能只有 `logo`，也可能两者都有）。
- 某次前向没有用到的分类头，其参数不参与该次 loss 与反向。DDP 默认要求「所有参数都参与
  反向」，否则会在下一个 iteration 报
  `Expected to have finished reduction in the prior iteration ...`；这条报错本身不直观，
  容易被误判成模型 bug。
- 开启后 DDP 会标记未使用参数并在该次反向中跳过它们，只同步真正参与计算的参数。
  示例配置已经写了 `true`，不要在多卡时删掉。
- 代价是每次迭代多一遍 autograd 图遍历。若某个 batch 恰好覆盖了全部任务，可能看到一条
  `find_unused_parameters=True was specified ... did not find any unused parameters` 的
  warning：这是「本 iteration 没有未使用参数」的提示，其他 batch 仍可能触发，**不要因此关掉该开关**。
- 单卡（`strategy: none`）不构造 DDP，该字段无影响。

验证步骤的 `preds` / `targets` 会被任务补齐为**模型全部任务名**的键（本 batch 没出现的任务
用空张量占位），因此各 rank 的字典键一致，DDP `all_gather` 不会因任务组成不同而崩。

### 4.3 链路冒烟结果（不是模型效果结论）

用仓库配套的真实 `multicls` 数据（2458 个样本，按 `(task, label)` 分层切分为
train 1967 / val 491）跑 1 epoch 的实测：

| 配置 | 结果 |
|---|---|
| 单卡 | train 123 batches / val 31 batches，1 epoch **12.1s**；`val_licence_accuracy 0.9581`、`val_logo_accuracy 0.7899` |
| 2 卡 torchrun | `strategy: ddp`，每 rank **61/61** train batches，per-task 指标正常跨卡聚合 |

> ⚠️ 上表只用于证明「数据 → 路由前向 → 逐任务损失 → 指标 → `history.csv`」**链路跑通**。
> 1 epoch 的数值不构成模型效果结论，不能用于比较模型、选型或调参；正式结论需要完整训练与
> 独立验证集。

---

## 5. 指标与日志

### 5.1 命名规则

| 阶段 | 任务返回的键 | `history.csv` 列 |
|---|---|---|
| 训练（每个 batch） | `loss`、`<task>_loss`、`<task>_acc` | `train_loss`、`train_<task>_loss`、`train_<task>_acc` |
| 验证（每个 epoch） | `loss`、`<task>_accuracy`、`<task>_precision`、`<task>_recall`、`<task>_f1` | `val_loss`、`val_<task>_accuracy` 等 |

- `<task>` 就是 `model.heads` 的键（也就是 `batch["task"]` 与 spec 的 `tasks[*].name`），
  例如 `licence` / `logo`。
- 训练键**随本 batch 实际出现的任务变化**：某个 batch 只有 `licence` 时，该 batch 的日志里
  就没有 `logo_loss` / `logo_acc`。epoch 级 `history.csv` 汇总本 epoch 出现过的全部键。
- 验证指标在任务**第一次出现**时按该任务的 logits 宽度惰性创建（分类预设：
  accuracy + macro precision / recall / f1），epoch 结束时返回 `<task>_<metric>` 并 reset。
  验证 split 中从未出现的任务不会产生对应的 `val_<task>_*` 列。

### 5.2 `history.csv` 列示例

真实 `multicls` 单卡 1 epoch 的实际表头（列顺序是字典序）：

```text
train_licence_acc,train_licence_loss,train_logo_acc,train_logo_loss,train_loss,
val_licence_accuracy,val_licence_f1,val_licence_precision,val_licence_recall,
val_logo_accuracy,val_logo_f1,val_logo_precision,val_logo_recall,val_loss
```

对应的一行（数值四舍五入，仅示意格式；上面已说明这是链路冒烟数字）：

```csv
train_licence_acc,train_licence_loss,train_logo_acc,train_logo_loss,train_loss,val_licence_accuracy,val_licence_f1,val_licence_precision,val_licence_recall,val_logo_accuracy,val_logo_f1,val_logo_precision,val_logo_recall,val_loss
0.8760,0.3730,0.8407,0.3384,0.7114,0.9581,0.6489,0.6436,0.6553,0.7899,0.7897,0.8221,0.8304,0.3710
```

### 5.3 损失的加权口径

- 每个任务用自己的样本子集算交叉熵；`<task>_loss` 记录该任务**未加权**的交叉熵。
- 总损失是各任务交叉熵按 `task_loss_weights` 加权后的**和**：
  `loss = Σ weight(task) × CE(task)`；未配置权重时权重均为 1.0。
- 验证步的 `val_loss` 与训练步采用**同一加权口径**，因此早停 / 调度器监视的
  `val_loss` 与 `train_loss` 可比。
- `train_<task>_loss` 与 `train_<task>_acc` 是每个 batch 的标量在 epoch 内由 Trainer
  按 **batch size** 加权平均得到的（Trainer 通用口径），**不是**该任务自己的严格样本级汇总；
  `val_<task>_accuracy` 则是在该任务全部验证样本上统计的。
- 分类预设的 precision / recall / f1 默认是 `average: macro`，可通过
  `metrics` 配置改（见 [指标](metrics.md)）。

---

## 6. 契约：模型、数据、任务

### 6.1 模型 `MultiTaskClassifier`

| 属性 | 形态 | 说明 |
|---|---|---|
| `supported_tasks` | 类属性 `("multi_task_classification",)` | 供 `training.contract` 的能力校验读取 |
| `task_names` | 只读 `tuple[str, ...]` | `heads` 的键，顺序稳定（dict 插入序） |
| `num_classes` | 只读 `dict[str, int]` | `{任务: head.num_classes}`；**不是**整数，也不能赋值 |
| `forward(x, task=None)` | `dict[str, Tensor]` | `task=None` 时每个头对整批算；给 `task` 时只对属于该任务的子集调用对应头 |

```python
from dvisionix.models import MultiTaskClassifier

model = MultiTaskClassifier(
    backbone={"type": "mobilenetv3_backbone"},
    heads={"licence": {"num_classes": 3}, "logo": {"num_classes": 2}},
)
print(model.task_names)    # ('licence', 'logo')
print(model.num_classes)   # {'licence': 3, 'logo': 2}
model(images)              # {licence: (B,3), logo: (B,2)}
model(images, task=["licence", "logo"])   # 路由：各任务只返回自己的子集
```

构建期会报错的情况：`heads` 为空、任务名为空或含 `.`、某个头缺 `num_classes`、
骨干既无 `num_features` 也无 `out_channels`（均抛 `ConfigError`）。

### 6.2 数据集 `DVXDataset`

多任务规格下：

| 属性 | 值 |
|---|---|
| `data_type` | `"multi_task_classification"` |
| `classes` / `num_classes` | `None`（没有统一类表） |
| `task_names` | `list[str]`，spec 的任务名 |
| `task_num_classes` | `dict[str, int]`，`{任务: len(classes)}`；即使传 `task="licence"` 过滤样本，该表仍是完整任务表 |
| `__getitem__` | `label`（int，该任务内 id）+ `task`（str） |

```python
from dvisionix.data import DVXDataset

ds = DVXDataset("./data/multicls/dvx_dataset.yaml", split="train")
print(ds.task_names)        # ['licence', 'logo']
print(ds.task_num_classes)  # {'licence': 3, 'logo': 2}
print(ds.num_classes)       # None
```

### 6.3 任务 `MultiTaskClassificationTask`

```python
task_type = "multi_task_classification"
accepted_model_tasks = ("multi_task_classification",)
accepted_data_types = ("multi_task_classification",)
required_batch_fields = ("image", "label", "task")
```

- `batch["task"]` 必须是 `list[str]`（长度等于 batch size）；`DataLoader` 的默认 collate
  会把每个样本的 `str` 拼成 `list[str]`，所以多任务数据源不能用会丢掉该字段的 collate。
- 训练入口 `tools/train.py` 在 `task_type: multi_task_classification` 时**不注入**
  `task.num_classes`，也不要求配置里有 `model.num_classes`；类别数由 heads 自带。
- `TASK_TYPES` 与 `_TASK_MAPPING` 都已包含 `multi_task_classification`（5 个训练任务类型：
  `classification` / `detection` / `segmentation` / `multi_task_classification` / `simclr`）。

### 6.4 三级校验中的字典类别数

`check_task_model` 在多任务场景走**逐任务比对**分支（与单任务的标量逻辑互斥）：

```text
dataset.task_num_classes = {licence: 3, logo: 2}
model.num_classes        = {licence: 3, logo: 2}    → 通过
```

任务名集合不一致（模型缺头 / 多头）或任一任务的类别数不一致时抛
`TaskDataMismatchError`，消息会一次列出缺失、多余与不一致的任务：

```text
数据集 DVXDataset 与模型 MultiTaskClassifier 的逐任务类别数不一致：
任务 'logo' 的类别数不一致（数据集 2，模型 3）。
提示：请让 model.heads 的键与 dvx_dataset.yaml 的 tasks 一一对应，
且每个分类头的 num_classes 等于该任务 classes 的长度。
```

任务 ↔ 模型能力不匹配则抛 `TaskModelMismatchError`。例如把多任务模型配给普通分类任务：

```text
任务 ClassificationTask 与模型 MultiTaskClassifier 不匹配：模型声明的任务能力为
['multi_task_classification']，任务要求的能力为 ['classification']，两者没有交集。
```

完整的校验时机、渐进式规则与 8 个内置任务契约见
[训练契约与自适应训练](training_contracts.md)。

---

## 7. 已知限制

如实列出，选择方案前请先确认不踩：

1. **margin 头暂不支持**。ArcFace / CosFace / SphereFace / AdaFace / NormFace /
   CurricularFace / Circle 等 margin 头依赖训练时的 `labels` 来施加角度/余弦 margin；
   开启 `num_sample_classes` 类别采样的 PartialFC 也需要 `labels` 才能工作。而路由前向
   `MultiTaskClassifier.forward(x, task=...)` **不会把 labels 透传给 head**：强行配置时
   margin 头**不会报错**，但 margin 静默失效、退化成普通余弦/线性 logits（PartialFC 则退化
   为不采样）；这比报错更危险，请勿在多任务 head 里使用这些类型（默认 `cls_head` 不受影响）。
2. **`synthetic` 数据源不支持多任务**。`data.dataset: synthetic` / `custom` 的合成数据没有
   `task` 字段，`build_synthetic_dataset` 对 `multi_task_classification` 会直接抛
   `ValueError`。请使用 `data.dataset: dvx` + 多任务规格。
3. **跨任务共享标签空间未实现**。每个任务一个独立 head、独立类别表；即使两个任务的类别
   完全相同，也不会共享分类器参数或标签映射。
4. **检索训练未实现**。多任务只覆盖分类，不提供 embedding / 检索 / re-ID 的 spec 字段、
   损失与评估。
5. **P3 不含推理/导出专项**。`model(images)`（`task=None`）会返回全部头的输出，但训练入口
   没有多任务推理/导出流程，多任务下 dict 输出的推理/导出**未做专项验证**；不要据此认为
   推理、导出已完成。
6. **DDP 仅验证单机多卡**，多机（多节点）未验证。`find_unused_parameters: true` 会带来
   额外的 autograd 图遍历开销；单卡没有收益也没有影响。
7. **训练侧 epoch 指标是 batch size 加权平均**，不是每个任务自己的样本级汇总（见 §5.3）；
   任务样本比例在 batch 间波动较大时，`train_<task>_*` 会有轻微口径偏差。验证侧指标是在
   该任务全部验证样本上统计的（DDP 下为全局聚合）。
8. **一张图片只能属于一个任务**。多任务目录之间不能重叠（同一图片被两个任务扫到会报
   `AnnotationError`）。若业务上需要「一张图同时打多套标签」，需要先改数据层与模型契约。
9. **训练 batch 没有任何可识别任务时**，任务会发出一次 `UserWarning` 并返回可反传的零损失，
   避免训练循环崩溃；这通常意味着数据任务名与模型 heads 不一致，日志里 `loss` 恒为 0
   就是信号。用 DVX 规格 + `tools/train.py` 时，这类不一致通常会先被 §6.4 的契约校验拦下。

---

## 8. 测试与验收

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python

# 全量测试（P3 完成后实测 963 passed / 1 skipped / 0 failed）
$PY -m pytest tests -q

# 多任务模型 / 任务 / 契约 / 装配
$PY -m pytest tests/test_models/test_multi_task.py \
  tests/test_training/test_multi_task_task.py \
  tests/test_training/test_contracts.py \
  tests/test_training/test_dvx_config_smoke.py -q

# 官方配置端到端门禁（configs/examples/ 下的外部数据模板不在其范围内）
$PY -m pytest tests/test_e2e_configs.py -q
```

覆盖点：

- 路由正确性：每个任务只对子集调用头、骨干只前向一次、`task=None` 时输出全部头；
- `task_names` 顺序稳定、`num_classes` 是 dict 且只读；
- 训练步逐任务 loss/acc、总损失按权重求和；验证步 `preds`/`targets` 键全集与空占位；
- 多任务指标按任务前缀惰性创建、跨 epoch 复用与 reset；
- `task_num_classes` 与模型字典类别数的缺失 / 多余 / 不一致均报错；
- DVX 多任务装配：`batch["task"]` 为 `list[str]`，模型能按 task 路由；
- 单卡 1 epoch 的 `history.csv` 同时包含两个任务的训练与验证指标。
