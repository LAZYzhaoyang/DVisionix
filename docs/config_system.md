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

## v1.1 变更要点

> 完整清单见 [v1.1 变更与迁移指南](v1.1_changes.md)。

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
