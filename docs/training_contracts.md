# 训练契约与自适应训练

> 适用范围：v2.0.0（P2/P3 契约）。本页面向使用 `tools/train.py`
> 或编程式 `Trainer` 的使用者，说明训练开始前的**任务 / 模型 / 数据**契约校验、
> 8 个内置任务分别接受什么、如何为自定义任务与模型声明能力，以及
> `strategy: auto` / `device: auto` / AMP 的当前行为。
>
> 接口定义以 [P2 实现契约](history/design/v2_p2_contract.md) 与
> [P3 实现契约](history/design/v2_p3_contract.md) 为准，本页只描述实际行为。
> 相关文档：[DVX 标准数据集](dataset_spec.md)、[快速开始](quick_start.md)、
> [配置系统](config_system.md)、[自定义模型](custom_models.md)、
> [路由式多任务分类](multi_task.md)。

---

## 1. 契约解决什么问题

P2 之前，下面这些错配往往要等模型前向、损失计算甚至训练到一半才暴露，
有的还会静默算错：

| 错配 | 旧表现 | 现在 |
|---|---|---|
| 分割任务配分类模型 | 形状/通道错误，堆栈难读 | 训练开始前抛 `TaskModelMismatchError` |
| 分类任务配 `data_type: detection` 的 DVX 数据 | 等到 batch 才缺字段 | 装配后、训练前抛 `TaskDataMismatchError` |
| 检测任务但 collate 没产出 `boxes` / `labels` | 第一次前向才报 KeyError | 第一个训练 batch 前抛 `TaskDataMismatchError` |
| `model.num_classes` 与 `dvx_dataset.yaml` 的 `classes` 数量不一致 | 标签越界或 loss 形状错误 | 训练前抛 `TaskDataMismatchError` |
| 多任务 `model.heads` 的任务名 / 类别数与 spec 的 `tasks` 不一致 | 训练到某个任务才因标签越界失败 | 训练前抛 `TaskDataMismatchError`，一次列出差异 |

这些错误都继承 `DVisionixError`，消息统一为「发生了什么 + 提示：如何修复」，
可以直接按提示改配置或改声明。

---

## 2. 三级契约校验

### 2.1 触发时机

```text
config → task → transforms(task) → dataset → model → trainer
                                                         │
trainer.fit(model)                                       │
  ├─ check_task_model(task, model, train_dataset)  ← 在 _wrap_model 之前
  ├─ check_task_data(task, train_dataset)           ← 训练开始前
  └─ 第一个训练 batch → check_batch_fields(task, batch) ← 第一次前向/反向之前
```

| 级别 | 校验函数 | 检查内容 | 失败异常 | 触发时机 |
|---|---|---|---|---|
| 任务 ↔ 模型 | `check_task_model` | 模型能力与 `accepted_model_tasks` 是否有交集；数据集与模型的类别数是否一致（单任务比较整数，多任务比较 `{任务: 类别数}` 字典） | `TaskModelMismatchError` / `TaskDataMismatchError` | `Trainer.fit()` 开始处，早于 DDP / `torch.compile` 包装 |
| 任务 ↔ 数据 | `check_task_data` | DVX 数据集的 `data_type` 是否在 `accepted_data_types` 内 | `TaskDataMismatchError` | 同上 |
| batch ↔ 任务 | `check_batch_fields` | 训练 batch 是否为 dict、是否含 `required_batch_fields` | `TaskDataMismatchError` | 整个 `fit()` 的**第一个训练 batch**，只检查一次 |

两个模型 / 数据校验都在优化器、DDP 包装和第一次前向之前完成；失败时
`trainer.model` 仍为 `None`、`global_step` 仍为 0，不会留下半初始化的训练状态。

### 2.2 任务 ↔ 模型能力

模型声明的能力由 `dvisionix.models.base.model_task_types(model)` 计算，优先级为：

1. `model.supported_tasks` 非空 → 使用它（可声明多个任务能力）；
2. 否则 `model.task_type` 非空 → 包装成单元素元组；
3. 两者都没有 → `()`（表示模型没有声明完整任务能力）。

`check_task_model` 取模型能力与 `task.accepted_model_tasks` 的交集；交集为空即报错。
例如把 `SegmentationTask` 配给一个只声明 `classification` 的 `SimpleCNN`：

```text
任务 SegmentationTask 与模型 SimpleCNN 不匹配：模型声明的任务能力为 ['classification']，
任务要求的能力为 ['segmentation']，两者没有交集。
提示：请让模型支持 ['segmentation'] 之一（设置模型的 task_type 或 supported_tasks），
或改用与模型能力 ['classification'] 匹配的任务；并检查配置中的 task.type 与 model.type 是否配对。
```

同一函数还会检查类别数：当**数据集与模型都暴露 `num_classes`** 且不相等时报
`TaskDataMismatchError`（与任务是否声明能力无关；传入数据集时执行）：

```text
数据集 DVXDataset 的 num_classes=2 与模型 SimpleCNN 的 num_classes=3 不一致。
提示：请让配置中的 model.num_classes 等于数据集类别数（分类/分割的类别表通常来自
dvx_dataset.yaml 的 classes），或确认 data.spec 指向了正确的数据集。
```

多任务分类没有全局类别数，走的是**字典类别数**分支：`DVXDataset.task_num_classes`
是 `{任务: 类别数}`，多任务模型的 `num_classes` 也是同样的字典时，逐任务比较
（任务名集合一致 + 每个任务的类别数一致）。缺失、多余或不一致会一次性列出：

```text
数据集 DVXDataset 与模型 MultiTaskClassifier 的逐任务类别数不一致：
任务 'logo' 的类别数不一致（数据集 2，模型 3）。
提示：请让 model.heads 的键与 dvx_dataset.yaml 的 tasks 一一对应，
且每个分类头的 num_classes 等于该任务 classes 的长度。
```

一边是逐任务字典、另一边是整数（例如多任务模型配单标签数据集）时同样报
`TaskDataMismatchError`，提示改用结构匹配的数据集 / 模型。多任务完整用法见
[路由式多任务分类](multi_task.md)。

修复方向：

- 换模型：让模型声明任务需要的能力（见 §5），或改用与模型匹配的任务；
- 核对配置：`task.type` 与 `model.type` 是否配对；
- 核对类别数：单任务让 `model.num_classes`、示例配置里的 `task.num_classes` 与
  `dvx_dataset.yaml` 的 `classes` 数量保持一致（分割通常含 `__background__`）；
  多任务让 `model.heads` 的键与 `tasks[*].name` 一一对应，且每个头的
  `num_classes` 等于该任务 `classes` 的长度。

### 2.3 任务 ↔ 数据 `data_type`

`DVXDataset` 暴露 `data_type`；`check_task_data` 要求它属于任务的
`accepted_data_types`。没有 `data_type` 属性的数据集（`CustomDataset`、公开数据集预设）
**直接放行**，以兼容 P2 之前的数据接入路径。

```text
数据集 DVXDataset 的 data_type='classification' 与任务 DetectionTask 不匹配：
任务接受的数据类型为 ['detection']。
提示：请检查 data.spec 指向的 dvx_dataset.yaml 的 data_type，或改用接受 ['detection']
的任务；分类 / 检测 / 分割任务分别对应 ClassificationTask / DetectionTask / SegmentationTask。
```

DVX 规格的 `data_type` 与内置任务的大致对应关系：

| `dvx_dataset.yaml` 的 `data_type` | 可接受的内置任务 |
|---|---|
| `classification` | `ClassificationTask`、`LinearEvalTask` |
| `detection` | `DetectionTask` |
| `segmentation` | `SegmentationTask`、`MaskFormerTask` |
| `multi_task_classification` | `MultiTaskClassificationTask`（配套模型 `MultiTaskClassifier`，见 [路由式多任务分类](multi_task.md)） |

> 内置任务的 `accepted_data_types` 还包含 `multi_label` / `simclr` /
> `instance_segmentation` / `panoptic_segmentation` 等值，但当前 DVX 规格只产生上表
> 四种 `data_type`；多标签、SimCLR、实例/全景分割训练需要相应的数据源与任务配置，
> 不要仅凭任务声明就认为已有现成的 DVX 训练链路。

### 2.4 batch 必需字段

`check_batch_fields` 只在第一个训练 batch 上执行一次。batch 必须是 dict，且包含任务
声明的所有 `required_batch_fields`：

```text
训练 batch 缺少任务 ClassificationTask 的必需字段 ['label']；实际 batch 键为 ['image']。
提示：请检查 data.dataset / data.spec 的 data_type 是否与任务一致，并确认 transforms
与 collate_fn 会产出这些字段（检测需 boxes/labels、分割需 mask、SimCLR 需 image1/image2）。
```

常见修复：

- `data_type` 写错（例如分类 spec 配检测任务）→ 修正 spec 或任务；
- transforms 没把标注转成张量 / 没写进 batch → 检查 `task.build_transforms` 返回的流水线；
- 检测 / 分割的 `DataLoader` 没传 `collate_fn=ds.collate_fn` → 用 `tools/train.py`
  时入口会自动使用 `getattr(ds, "collate_fn", None)`，编程式路径需显式传入。

### 2.5 渐进式契约

任务若**没有声明** `accepted_model_tasks` / `accepted_data_types`（保持 `BaseTask`
的默认空元组），对应的校验会被跳过，而不是拿空集合去和模型/数据比较：

- 未声明 `accepted_model_tasks` → 跳过任务 ↔ 模型能力校验；
- 未声明 `accepted_data_types` → 跳过 `data_type` 校验；
- `num_classes` 一致性校验**不受影响**，仍会执行（传入数据集时）；
- `required_batch_fields` 默认是 `("image",)`，可用子类覆盖。

内置 8 个任务均已显式声明，所以「分割任务配分类模型」仍会在训练开始前被拦下。
渐进式规则只用于兼容尚未补声明的自定义任务；新任务建议主动声明，避免把错配留到运行期。

---

## 3. 8 个内置任务的契约表

以下以代码中的类属性为准（与 `tests/test_training/test_contracts.py` 的契约表一致）：

| Task 类 | `task_type` | `accepted_model_tasks` | `accepted_data_types` | `required_batch_fields` |
|---|---|---|---|---|
| `ClassificationTask` | `classification` | `("classification",)` | `("classification",)` | `("image", "label")` |
| `MultiLabelTask` | `multi_label` | `("classification",)` | `("multi_label",)` | `("image", "label")` |
| `DetectionTask` | `detection` | `("detection",)` | `("detection",)` | `("image", "boxes", "labels")` |
| `SegmentationTask` | `segmentation` | `("segmentation",)` | `("segmentation",)` | `("image", "mask")` |
| `MaskFormerTask` | `segmentation` | `("segmentation",)` | `("segmentation", "instance_segmentation", "panoptic_segmentation")` | `("image", "mask")` |
| `SimCLRTask` | `simclr` | `("classification",)` | `("simclr",)` | `("image1", "image2")` |
| `LinearEvalTask` | `linear_eval` | `("classification",)` | `("classification",)` | `("image", "label")` |
| `MultiTaskClassificationTask` | `multi_task_classification` | `("multi_task_classification",)` | `("multi_task_classification",)` | `("image", "label", "task")` |

字段含义：

- `task_type`：任务自己的类型名，用于日志与报错；不是配置顶层的 `task_type`。
- `accepted_model_tasks`：任务可接受的**模型任务能力**，与模型的 `supported_tasks` /
  `task_type` 比较。
- `accepted_data_types`：任务可接受的 `DVXDataset.data_type`。
- `required_batch_fields`：训练 batch 必须携带的键；检测是变长的 `boxes` + `labels`，
  分割是 `mask`，SimCLR 是双视角 `image1` / `image2`，多任务是 `image` + `label` +
  按样本的 `task`（`list[str]`）。

关于配置入口的两个细节：

1. 配置顶层的 `task_type` 只能是 `classification` / `detection` / `segmentation` /
   `multi_task_classification` / `simclr`（共 5 个训练任务类型，权威来源是
   `dvisionix/config/schema.py::TASK_TYPES`）；`tools/train.py` 的缺省 `task.type`
   映射 `_TASK_MAPPING` 与它完全一致，也包含 `multi_task_classification`。
   使用 `MultiLabelTask` / `LinearEvalTask` / `MaskFormerTask` 时，请在配置里显式写
   `task.type`（例如 `task_type: classification` + `task.type: LinearEvalTask`）。
   其中 `MultiLabelTask` / `MaskFormerTask` 还需要能产出对应 batch 字段的数据源
   （DVX 规格当前不产生 `multi_label` / `instance_segmentation` /
   `panoptic_segmentation`，见 §2.3）。
2. 8 个任务都实现了 `build_transforms(train, image_size)`；配置驱动训练时由
   `tools/train.py` 向任务索取预设变换，任务返回 `None` 会直接抛 `ConfigError`
   （自定义任务需自行实现，见 §4）。

---

## 4. 自定义任务：声明能力

自定义任务继承 `BaseTask`，至少实现 `training_step` / `validation_step`；若要走
`tools/train.py` 的配置驱动路径，还需要注册到 `TASKS` 并提供 `build_transforms`。

下面是一个可运行的最小分类任务（与内置 `ClassificationTask` 的关键差别是没有指标
与额外日志，只保留训练/验证所需字段）：

```python
import torch
import torch.nn.functional as F

from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.registry import TASKS
from dvisionix.training import BaseTask


@TASKS.register()  # 以类名 "MyClassificationTask" 注册
class MyClassificationTask(BaseTask):
    # 任务契约：声明自己接受的模型能力、数据类型与 batch 字段
    task_type = "classification"
    accepted_model_tasks = ("classification",)
    accepted_data_types = ("classification",)
    required_batch_fields = ("image", "label")

    def __init__(self, num_classes=None, **kwargs):
        # tools/train.py 会传入 num_classes / optimizer_cfg / scheduler_cfg / loss / metrics
        super().__init__(**kwargs)
        self.num_classes = num_classes

    @classmethod
    def build_transforms(cls, train: bool, image_size: int):
        """配置驱动训练时，入口通过这里获取该任务的预设变换。"""
        return ClassificationTransforms(train=train, image_size=image_size)

    def training_step(self, model, batch, device):
        images = self.to_device(batch["image"], device)
        labels = self.to_device(batch["label"], device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss}

    def validation_step(self, model, batch, device):
        images = self.to_device(batch["image"], device)
        labels = self.to_device(batch["label"], device)
        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "preds": logits, "targets": labels}
```

声明要点：

- **必须声明** `accepted_model_tasks` / `accepted_data_types` 才能让对应校验生效；
  不声明就按 §2.5 跳过，属于渐进式兼容。
- `required_batch_fields` 必须覆盖 `training_step` 真正读取的键，否则会在第一个
  batch 前被校验拦下。
- 想复用内置任务的契约，可以直接继承对应任务，只覆盖需要改的步骤，例如
  `class MyTask(ClassificationTask): ...`。
- `build_transforms` 返回 `None` 时，`tools/train.py` 会报
  `ConfigError: ... 未提供预设变换`；编程式传给 `Trainer` 的 DataLoader 不受此限制。

配置写法示例：

```yaml
task_type: classification        # 顶层是 5 个训练任务类型之一
task:
  type: "MyClassificationTask"   # 注册名（类名），必须已在当前进程 import/注册
  num_classes: 5
```

> `tools/train.py` **不会自动发现**未导入的自定义任务模块。若通过 YAML 使用自定义任务，
> 需要在自己的启动脚本中先 `import` 该模块（触发 `@TASKS.register()`），再调用入口逻辑；
> 或直接编程式构建 Task 后交给 `Trainer`。

---

## 5. 自定义模型：两种能力声明

`model_task_types` 支持两种声明方式，**`supported_tasks` 优先于 `task_type`**。

### 方式一：`task_type`（单任务模型）

```python
import torch.nn as nn

from dvisionix.models import BaseModel
from dvisionix.registry import MODELS


@MODELS.register()
class MyClassifier(BaseModel):
    def __init__(self, num_classes=10, in_channels=3):
        # 推荐在构造时传入；它会被写入实例的 task_type
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, labels=None):
        # ClassificationTask 训练步会传 labels=；普通分类头可以忽略它
        return self.net(x)
```

### 方式二：`supported_tasks`（可声明多个能力）

```python
import torch.nn as nn

from dvisionix.models import BaseModel
from dvisionix.registry import MODELS


@MODELS.register()
class MyMultiAbilityModel(BaseModel):
    # 声明自己同时具备分类与分割能力；这不会让它自动变成多任务分类模型
    # （多任务训练需要 MultiTaskClassifier 的 heads 契约，见 multi_task.md）
    supported_tasks = ("classification", "segmentation")

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # ... 省略实际网络 ...

    def forward(self, x, labels=None):
        ...
```

> **注意**：`BaseModel.__init__` 会把实例的 `task_type` 设为传入值（默认 `None`）。
> 因此只写类属性 `task_type = "classification"`、又调用 `super().__init__()` 时，
> 类属性会被实例上的 `None` 覆盖，`model_task_types` 返回 `()`。
> 请使用 `super().__init__(task_type="classification")`，或在 `super().__init__()` 之后
> 赋值 `self.task_type = "classification"`。不是 `BaseModel` 的普通 `nn.Module` 写类属性
> 则不受影响。

补充：

- `num_classes` 是**可选**属性：模型与数据集都暴露它时才会做一致性校验；没有声明就跳过
  这一项（而不是默认不校验任务能力）。
- 与 `ClassificationTask` 搭配时，模型 / 头必须接受 `labels=` 关键字参数（可以忽略）。
  `ClsHead` / `MultiLabelHead` 已在 P2 补上该兼容，否则
  `LinearClassifier + ClsHead + ClassificationTask` 会在第一个 batch 报 `TypeError`。
- 配置中通过 `model.type: "MyClassifier"` 使用，同样需要保证类已在当前进程注册/导入。

---

## 6. 自适应训练

### 6.1 `strategy: auto`：识别 torchrun

`Trainer._init_distributed()` 的判定顺序：

1. 进程组已初始化且 `world_size > 1` → `ddp`；
2. 环境变量 `RANK` 与 `WORLD_SIZE` 同时存在，且 `int(WORLD_SIZE) > 1` → `ddp`
   （torchrun 会注入这些变量，此时进程组可能尚未初始化）；
3. 否则 → `none`（普通单进程启动，不会被误判为分布式）。

判定为 `ddp` 后，若进程组尚未初始化，由 Trainer 调用
`dist.init_process_group`：CUDA 且未显式指定 CPU 时用 `nccl`，否则用 `gloo`。
如果显式写 `strategy: ddp` 却没有进程组、也不是 torchrun 启动，会得到明确的
`RuntimeError`（提示用 torchrun 或先手动初始化）。

```bash
# 单机 2 卡；strategy 默认 auto，无需额外参数
torchrun --nproc_per_node=2 tools/train.py \
  --config configs/examples/dvx_classification.yaml \
  --cfg-options data.spec=/path/to/cls/dvx_dataset.yaml \
  --work-dir /tmp/dvx_cls_ddp
```

实测日志中，每个 rank 会打印 `Using device: cuda:0/1, amp: False, strategy: ddp`；
入口先看到完整的 `Train batches: 8`，Trainer 用 `DistributedSampler` 切分后每个 rank
训练 4 个 batch（`8 → 4/4`）。`--devices 1,3` 会按 `LOCAL_RANK` 映射到
`cuda:1` / `cuda:3`。

### 6.2 `device: auto`

`device: auto` 走通用设备解析：有 CUDA 用 CUDA，否则有 MPS 用 MPS，否则 CPU。
DDP + CUDA 时，每个进程会改为绑定自己的卡（`cuda:<LOCAL_RANK>`，或按
`--devices` / `training.devices` 的映射），而不是所有进程都用 `auto` 选到的同一张卡。
显式写 `training.device: cpu` 时 DDP 走 CPU gloo，不会绑定 GPU。

### 6.3 AMP：仍由配置显式开启

AMP **没有**自动开启逻辑，默认值就是关闭：

```yaml
training:
  amp: false        # 默认值；需要时显式改成 true
```

- 只有 `training.amp: true` 且实际设备是 CUDA 时才创建 `GradScaler` 并启用
  `torch.autocast`；
- CPU / MPS 上写 `amp: true` 不会启用混合精度（也不会因此报错）；
- 日志里的 `amp: False/True` 就是最终是否启用；
- 四个 DVX 示例配置都显式写了 `amp: false`。

---

## 7. 用 DVX 数据集训练

训练入口用 `data.dataset: dvx` + `data.spec` 构建 `DVXDataset`，并以任务的
`build_transforms` 作为变换来源；检测/分割的 `DataLoader` 会自动带上数据集自带的
`collate_fn`。完整命令与三类示例见 [快速开始](quick_start.md) 的
「用 DVX 标准数据集训练」，这里只给一个最小调用：

```bash
python tools/train.py \
  --config configs/examples/dvx_classification.yaml \
  --cfg-options data.spec=/path/to/cls/dvx_dataset.yaml \
  --work-dir /tmp/dvx_cls
```

- `data.spec` 指向 `dvx_dataset.yaml`；相对路径按**当前工作目录**解析，规格里的
  `root` 相对**规格文件所在目录**解析；
- 单任务训练的 `model.num_classes` 必须与 spec 的类别数一致，否则由 §2.2 的契约校验
  在训练前拦下；多任务训练没有全局 `model.num_classes`，改由 `model.heads` 与 spec 的
  `tasks` 逐任务比对，见 [路由式多任务分类](multi_task.md)；
- `configs/examples/dvx_{classification,detection,segmentation,multi_task}.yaml` 是训练
  配置模板，`data.spec` 为占位路径，使用前请替换；
- `data.dataset` 不写时默认 `custom`，仍保留合成数据与公开数据集预设路径。

---

## 8. 已知边界

- **多任务分类（P3）已可用，但有明确限制**：只支持路由式分类，margin 头（ArcFace /
  CosFace 等）因路由前向不透传 `labels` 而**暂不支持**（强行使用不会报错，但 margin 会
  静默失效）；`synthetic` 数据源不支持多任务；跨任务共享标签空间、检索训练、多任务推理/
  导出专项均未实现。详见 [路由式多任务分类](multi_task.md)。
  §5 的 `supported_tasks` 只表示模型可以声明多个能力，不等于多任务训练自动可用。
- **DDP 当前是单机多卡**。`torchrun --nproc_per_node=N` 的单机场景已验证；多机
  （多节点）DDP 未做专门适配与验证。
- **验证聚合先转 CPU**。多卡验证时 `preds` / `targets` 会先递归搬到 CPU 再
  `all_gather_object`，合并后搬回训练设备供 `task.update_metrics` 使用。这是为了规避
  跨设备 `torch.cat` 崩溃；代价是每次验证的一次 CPU 往返，`preds` 很大时通信与内存开销
  会上升。指标只在 rank0 上用全局结果计算，非 rank0 的 `val_*` 不作为训练结果。
- **`DVXDataset` 未注册到 `DATASETS`**。它通过 `data.dataset: dvx` + `data.spec` 的专用
  分支构建；`build_dataset({...})` 仍不能按名字直接构建它。
- **P3 仍不含推理与导出专项**。本页只覆盖训练入口、契约与自适应训练；多任务模型的
  推理/导出链路不在 P3 验收范围内，不要据此认为它们已完成。
- **契约校验有边界**：非 DVX 数据集跳过 `data_type` 校验；自定义任务未声明能力时跳过
  对应校验；`num_classes` 只在数据集与模型都暴露该属性时比较（多任务比较两个字典）。
  这些是渐进式兼容的取舍，不是“所有错配都能被拦下”的保证。

---

## 9. 相关测试

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python

# 契约定义与 Trainer 接线
$PY -m pytest tests/test_training/test_contracts.py -q

# DVX 训练入口装配（快速，不真正训练）
$PY -m pytest tests/test_training/test_dvx_config_smoke.py -q

# 多任务模型 / 任务 / 契约
$PY -m pytest tests/test_models/test_multi_task.py \
  tests/test_training/test_multi_task_task.py -q

# DDP：须由 torchrun 启动，且需要 2+ 张 CUDA 卡
torchrun --nproc_per_node=2 -m pytest tests/test_training/test_ddp_smoke.py -q
```
