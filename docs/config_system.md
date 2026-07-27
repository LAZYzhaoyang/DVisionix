# 配置系统

`dvisionix.config.Config` 提供统一的 YAML 配置管理，支持继承、深度合并、点号访问与验证。

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

## 合并 / 验证 / 保存

```python
merged = cfg.merge({"training": {"lr": 0.01}}, override=True)
cfg.validate(["task_type", "model.name", "training.num_epochs"])  # 缺字段抛 ValueError
cfg.dump("configs/_generated/effective.yaml")
```
