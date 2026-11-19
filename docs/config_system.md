# 配置系统

`dvisionix.config.Config` 提供统一的 YAML 配置管理，支持继承、深度合并、点号访问、CLI 覆盖与 schema 校验。

## 加载与访问

```python
from dvisionix.config import Config

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
print(cfg.model.name)            # 点号访问
print(cfg["training"]["batch_size"])  # 字典访问
print(cfg.get("experiment_name", "unnamed"))  # 安全访问
```

## 默认配置

内置各任务默认配置，位于 `dvisionix/config/defaults/`：

```python
cfg = Config.from_default("classification")  # classification | detection | segmentation
```

## 配置继承（`_base_`）

在 YAML 顶部用 `_base_` 指定父配置（支持相对路径与列表）。当前配置的字段会**深度合并**并覆盖父配置：

```yaml
_base_: "../../dvisionix/config/defaults/classification.yaml"
model:
  num_classes: 4          # 覆盖默认值
training:
  num_epochs: 2           # 仅覆盖该字段，其余保留
```

## CLI 覆盖

`tools/train.py --cfg-options a.b=v`，值支持 int / float / bool / null 与 YAML 子集（list / dict）：

```bash
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --cfg-options training.num_epochs=5 training.optimizer.lr=0.01 training.devices=[0,1]
```

## 合并 / 验证 / schema 校验 / 保存

```python
merged = cfg.merge({"training": {"optimizer": {"lr": 0.01}}}, override=True)
cfg.validate(["task_type", "model.num_classes", "training.num_epochs"])  # 缺字段抛 ValueError
warnings = cfg.validate_schema(cfg.task_type)  # 类型/取值校验 + 未知键/别名告警
for w in warnings:
    print(w)
cfg.dump("configs/_generated/effective.yaml")
```
---

## v2 配置要点

> 完整使用说明见 [训练契约与自适应训练](training_contracts.md) 与
> [DVX 标准数据集](dataset_spec.md)。

**5 个训练任务类型**（`task_type` / `task.type`；权威来源
`dvisionix/config/schema.py::TASK_TYPES`，训练入口与 schema 校验共用）：

| 训练任务类型 | 数据来源 | 说明 |
|---|---|---|
| `classification` | DVX `classification` / `CustomDataset` / 公开数据集预设 | 单标签分类（含度量学习头） |
| `detection` | DVX `detection` / 公开数据集预设 | 目标检测 |
| `segmentation` | DVX `segmentation` / 公开数据集预设 | 语义 / 实例分割 |
| `multi_task_classification` | DVX `multi_task_classification` | 路由式多任务分类 |
| `simclr` | `CustomDataset` / 合成数据 | 自监督对比学习（无 `data_type`，数据契约跳过） |

> 注意 `dvisionix/models/base.py::TASK_TYPES` 是**模型任务**类型
> （`classification` / `detection` / `segmentation`，不含 `simclr`），
> 与上面的训练任务类型是两个概念。

**DVX 数据集接线**：`data.dataset: dvx` + `data.spec`（指向 `dvx_dataset.yaml`）：

```yaml
data:
  dataset: "dvx"
  spec: "./data/cls/dvx_dataset.yaml"   # 必填；缺失时训练入口直接抛 ConfigError
  image_size: 224
```

**多任务分类没有全局 `model.num_classes`**：每个任务一个分类头，类别数写在
`model.heads` 内，且任务名集合、各任务类别数必须与 spec 的 `tasks` 一致：

```yaml
task_type: "multi_task_classification"
model:
  type: "multi_task_classifier"
  backbone: {type: "mobilenetv3_backbone", features_only: false}
  heads:
    licence: {type: "cls_head", num_classes: 3}
    logo:    {type: "cls_head", num_classes: 2}
```

**v2 新增 / 变更的 `training` 配置键**：

```yaml
training:
  find_unused_parameters: true    # DDP 下允许部分 head 参数未参与本 batch（多任务建议开启）
  early_stopping:
    restore_best_weights: true    # 2.0.1 起真正生效；false = 早停后保留最后一轮权重
  export_best_onnx: false         # 训练结束导出 work_dir/best.onnx 与 best.onnx.json（2.0.1 起）
```

- `find_unused_parameters` 默认 `false`，直接转发给 DDP 包装；路由式多任务下
  一个 batch 未必覆盖全部 head，开启后避免「未使用参数」报错。
- `restore_best_weights` 此前只在 schema 白名单里、并未传给 `EarlyStopping`，
  写了也静默按默认 `true` 执行；2.0.1 起真正生效。
- `export_best_onnx` 2.0.1 起复用 P5 导出体系（`build_exporter("onnx", ...)`），
  除 `best.onnx` 外还落盘 `best.onnx.json` sidecar；导出失败只告警，不影响训练。

**契约校验时机**：训练入口在构建数据 / 模型 / 任务之后、训练循环开始前校验
**任务↔模型能力**与**任务↔数据 `data_type`**；`batch` 必需字段在**第一个训练 batch、
首次前向之前**校验一次。错配会在训练开始前给出「发生了什么 + 如何修复」的领域异常。
详见 [训练契约与自适应训练](training_contracts.md)。

---

## v1.1 变更要点

> 完整清单见 [v1.1 变更与迁移指南](history/v1.1_changes.md)。

**新增 / 补登记的 `training` 配置键**（v1.0.0 里写了也不生效，或会打出
「未知 training 配置键」的误导性告警）：

```yaml
training:
  compile: false              # 补登记（v1.0.0 已在 base.yaml 中但 schema 未收录）
  channels_last: false        # 同上
  export_best_onnx: false     # 同上
  non_blocking: false         # 新增：CPU→GPU 异步搬运，需配合 pin_memory
  pin_memory: false           # 新增：锁页内存
  persistent_workers: false   # 新增：仅在 num_workers > 0 时生效
  prefetch_factor: null       # 新增
  ema:                        # 新增：EMA 此前只能编程式使用
    enabled: true
    decay: 0.999
    decay_warmup_epochs: 5
    swap_for_validation: true
    save_final: true
```

**Registry 选择器**：`type` 是标准写法；`name` 仍可用但会发出
`DeprecationWarning`（v1.0.0 的 6 个官方配置已迁移到 `type`）；
`_name_` 是 `type` 的显式别名，用于配置里同时存在构造参数 `name` 的场景。

```yaml
model:
  type: "timm_backbone"   # 推荐
  name: "resnet18"        # 这是构造参数，原样传给 TimmBackbone
```

**顶层导入改为惰性**：`from dvisionix.config import Config` 不再加载 torch
（约 3–4 s → 0.1 s）。代价是组件注册发生在子模块被导入时 ——
使用注册表前请显式 `import dvisionix.models`（或 `training` / `metrics`）。

---

## v1.1.1 变更要点

**`resume` 的三种写法**（`false` / `true` / `"auto"` / `"latest"` / 显式路径）：

```yaml
resume: false          # 全新训练（默认）
resume: true           # 等价于 "auto"：续最近一次任务
resume: "auto"         # 同上
resume: "latest"       # 同上
resume: "/path/to/last.pt"   # 显式 checkpoint 路径
```

`true` 此前会被**原样当成路径**返回，最终 `torch.load(True)` 抛
`AttributeError: 'bool' object has no attribute 'seek'`；而且工作目录解析也会把它当作
"非 auto"，于是**既没续训、又另起了一个带新时间戳的目录**。现在 `true` 等价于 `"auto"`，
其它非字符串值会直接抛带提示的 `ValueError`。

> ⚠️ `resume` 参与配置哈希计算。**在配置里新增** `resume: true/auto` 会改变哈希，
> 从而被续训一致性校验拒绝（提示 `配置哈希不一致`）。用命令行
> `--resume auto` 则不影响哈希，推荐这种用法。

**`training.devices` 现在真正生效**：`--devices 1,3` / `devices: [1,3]` 会依次映射到
local_rank 0、1 绑定 GPU 1 与 GPU 3；此前该键只被赋值、从未被读取。
另外显式 `device: "cpu"` 不再被 DDP 自动绑卡覆盖（用于在 GPU 机器上跑 CPU 回归）。
