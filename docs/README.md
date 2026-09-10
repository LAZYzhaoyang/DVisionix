# DVisionix 文档索引

> **刚升级到 v1.1？** 请先看 [v1.1 变更与迁移指南](v1.1_changes.md) ——
> 其中列出了会改变训练结果、输出格式或 API 行为的变更。

## 快速开始
- [快速开始](quick_start.md) - 安装、Config 驱动训练、编程式训练、训练工程增强
- [配置系统](config_system.md) - Config 加载、继承、CLI 覆盖、schema 校验
- [v1.1 变更与迁移指南](v1.1_changes.md) - 20 项缺陷修复、行为变更、新配置项、API 变化

## 核心模块
- [数据模块](data.md) - Sample 协议、BaseDataset、原子变换与裁剪契约、公开数据集工具箱
- [骨干网络](backbones.md) - 12 种内置骨干（CNN/Transformer/timm）与预训练加载
- [自定义 Layer 与 Model](custom_models.md) - layers 模块、注册表与配置驱动组装
- [指标 (Metrics)](metrics.md) - 原子指标、组合容器、预设与自定义
- [日志系统](logging.md) - 结构化日志 / JSONL / TensorBoard（TrainingLogger）
- [模型导出 (ONNX)](model_export.md) - ONNXExporter 导出与精度验证
- [语义分割任务](segmentation.md) - 分割数据格式与端到端训练
- [目标检测任务](detection.md) - 组件化检测器、decode 契约、损失权重与 mAP 评估

## 工具与入口
- `tools/train.py` - 配置驱动统一训练入口（分类/检测/分割/自监督，支持 resume / DDP / work_dir / ONNX 导出）
- `tools/hparam_search.py` - 超参搜索（参数网格/随机采样，逐 trial 独立进程）
- `tools/benchmark.py` - 性能基准（EMA / PQ / mAP / matcher / assigner / 数据加载 / 导入开销）

## 说明
- 项目规划与开发约束见 [CodePlan](../CodePlan.md)；使用文档以本目录为准。
- 变更历史见 [CHANGELOG](../CHANGELOG.md)。
- 运行任何脚本前请先激活 conda 环境 `dvisionix`（Windows 上尤其重要：
  未激活时 `Library\bin` 不在 `PATH`，子进程 `import torch` 会报 `DLL load failed`）。
- 测试：`pytest tests/`；只要快速反馈时用 `pytest tests -m "not slow"`。
