# DVisionix 文档索引

> **当前版本 v2.0.1**（2026-11-20）。v2 系列带来：DVX 标准数据集层、训练任务/模型/数据
> 三级契约、路由式多任务分类、统一推理链路（`PreprocessSpec` / `Results` / `Predictor` /
> `tools/infer.py`）与可扩展导出体系（`BaseExporter` / `EXPORTERS` / `tools/export.py`）。
> 2.0.1 清理了废弃的 `use_giou`，接通 `training.early_stopping.restore_best_weights`，
> 并让 `training.export_best_onnx` 复用导出体系（额外落盘 `best.onnx.json`）。
>
> ⚠️ **升级提示**：v2.0.0 含接口与行为变更（`ONNXExporter.export/verify` 返回类型、
> 训练前契约校验拦截错配、`ClsHead` 开始接受 `labels` 等），升级前请先读
> [CHANGELOG 2.0.0](../CHANGELOG.md) 与 [2.0.1](../CHANGELOG.md) 章节。

## 快速开始
- [快速开始](quick_start.md) - 安装、Config 驱动训练、编程式训练、训练工程增强
- [配置系统](config_system.md) - Config 加载、继承、CLI 覆盖、schema 校验与 v2 配置要点
- [训练契约与自适应训练](training_contracts.md) - 任务/模型/数据三级校验、8 个内置任务契约、自定义能力声明、`strategy: auto` / `device` / AMP

## 数据与任务
- [DVX 标准数据集](dataset_spec.md) - 一个根目录 + 一份 dvx_dataset.yaml；字段、三种切分、DVXDataset、训练接线与四套真实数据
- [路由式多任务分类](multi_task.md) - 共享骨干 + 每任务一个分类头、`multi_task_classification` 数据、配置/训练/DDP 与已知限制
- [目标检测任务](detection.md) - 组件化检测器、decode 契约、损失权重与 mAP 评估
- [语义分割任务](segmentation.md) - 分割数据格式、DVX 真实数据与端到端训练
- [数据模块](data.md) - Sample 协议、BaseDataset、原子变换与裁剪契约、公开数据集工具箱、DVX 数据集

## 模块参考
- [推理链路](inference.md) - 预处理规格、统一 Results、三任务 Predictor、CLI 与已知限制
- [模型导出 (ONNX)](model_export.md) - BaseExporter/EXPORTERS/ExportResult 导出体系、`tools/export.py` CLI 与 ONNX Runtime 一致性校验（TensorRT 仅扩展点）
- [骨干网络](backbones.md) - 12 种内置骨干（CNN/Transformer/timm）与预训练加载
- [自定义 Layer 与 Model](custom_models.md) - layers 模块、注册表、配置驱动组装与模型能力声明
- [指标 (Metrics)](metrics.md) - 原子指标、组合容器、预设与自定义
- [日志系统](logging.md) - 结构化日志 / JSONL / TensorBoard（TrainingLogger）

## 工程工具
- `tools/train.py` - 配置驱动统一训练入口（分类/检测/分割/多任务/自监督，支持 DVX 数据集、resume / DDP / work_dir / ONNX 导出）
- `tools/infer.py` - 配置驱动推理入口（分类/检测/分割，输出 JSON 与可视化）
- `tools/export.py` - 配置驱动导出入口（ONNX + `<output>.json` sidecar + 一致性校验；TensorRT 仅扩展点）
- `tools/hparam_search.py` - 超参搜索（参数网格/随机采样，逐 trial 独立进程）
- `tools/benchmark.py` - 性能基准（EMA / PQ / mAP / matcher / assigner / 数据加载 / 导入开销）

## 架构与约束
- [架构与分层规则](architecture.md) - R1-R3、layers.yaml、check_layering 用法、检查范围与例外登记流程

## 历史归档
> 以下为历史记录，**非当前使用文档**；其配置项与测试数字反映归档时的状态。

- [历史归档说明](history/README.md) - 目录用途、归档内容与使用方式
- [v1.1 变更与迁移指南](history/v1.1_changes.md) - 20 项缺陷修复、行为变更、新配置项、API 变化
- [v1.1 稳定性与工程优化计划](history/v1.1_plan.md) - 已完成的 v1.1 / v1.1.1 执行计划与验收记录
- [v2 实现契约（P0–P5）](history/design/) - v2 各阶段的接口实现契约

## 说明
- 项目规划与开发约束见 [CodePlan](../CodePlan.md)；当前使用文档以本目录为准。
- 变更历史见 [CHANGELOG](../CHANGELOG.md)。
- 运行任何脚本前请先激活 conda 环境 `dvisionix`（Windows 上尤其重要：
  未激活时 `Library\bin` 不在 `PATH`，子进程 `import torch` 会报 `DLL load failed`）。
- 测试：`pytest tests/`；只要快速反馈时用 `pytest tests -m "not slow"`。
