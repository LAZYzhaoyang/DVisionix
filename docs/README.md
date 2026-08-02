# DVisionix 文档索引

## 快速开始
- [快速开始](quick_start.md) - 安装、Config 驱动训练、编程式训练、训练工程增强（P1-P3）
- [配置系统](config_system.md) - Config 加载、继承、CLI 覆盖、schema 校验

## 核心模块
- [数据模块](data.md) - Sample 协议、BaseDataset、原子变换、公开数据集工具箱
- [骨干网络](backbones.md) - 12 种内置骨干（CNN/Transformer/timm）与预训练加载
- [自定义 Layer 与 Model](custom_models.md) - layers 模块、注册表与配置驱动组装
- [指标 (Metrics)](metrics.md) - 原子指标、组合容器、预设与自定义
- [日志系统](logging.md) - 结构化日志 / JSONL / TensorBoard（TrainingLogger）
- [模型导出 (ONNX)](model_export.md) - ONNXExporter 导出与精度验证
- [语义分割任务](segmentation.md) - 分割数据格式与端到端训练
- [目标检测任务](detection.md) - 组件化检测器、decode 与 mAP 评估

## 工具与入口
- `tools/train.py` - 配置驱动统一训练入口（分类/检测/分割/自监督，支持 resume / DDP / work_dir / ONNX 导出）
- `tools/hparam_search.py` - 超参搜索（参数网格/随机采样，逐 trial 独立进程）

## 说明
- 项目规划与开发约束见 [CodePlan](../CodePlan.md)；使用文档以本目录为准。
- 运行任何脚本前请先激活 conda 环境 `dvisionix`；测试使用 `pytest tests/`。
