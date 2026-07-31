# NOTE: v0.2.0 起本仓库已迁移到 pytest。请使用:
#   pytest tests/
# 该脚本仅作为历史保留，最新覆盖等价于 tests/test_data/ 与 tests/test_models/。
# D:\\ZhaoyangProject\\DVisionix\\verify_all_modules.py

"""
验证所有模块是否能正常导入
"""

import sys
sys.path.insert(0, "D:\\ZhaoyangProject\\DVisionix")

print("=" * 60)
print("DVisionix 模块导入验证")
print("=" * 60)

def test_import(name, module_path):
    try:
        __import__(module_path)
        print(f"[OK] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False

all_passed = True

# 核心模块
print("\n核心模块:")
all_passed &= test_import("dvisionix", "dvisionix")
all_passed &= test_import("dvisionix.data", "dvisionix.data")
all_passed &= test_import("dvisionix.models", "dvisionix.models")
all_passed &= test_import("dvisionix.training", "dvisionix.training")
all_passed &= test_import("dvisionix.metrics", "dvisionix.metrics")
all_passed &= test_import("dvisionix.utils", "dvisionix.utils")

# 数据适配器
print("\n数据适配器:")
all_passed &= test_import("BaseDataset", "dvisionix.data.base")
all_passed &= test_import("CustomDataset", "dvisionix.data.datasets.custom")
all_passed &= test_import("CIFAR10Dataset", "dvisionix.data.presets")
all_passed &= test_import("CocoDetectionDataset", "dvisionix.data.presets")

# 数据变换
print("\n数据变换:")
all_passed &= test_import("BaseTransform", "dvisionix.data.transforms.base")
all_passed &= test_import("ImageResize", "dvisionix.data.transforms.image")
all_passed &= test_import("BoxSyncResize", "dvisionix.data.transforms.geometric")

# 训练模块
print("\n训练模块:")
all_passed &= test_import("Trainer", "dvisionix.training.trainer")
all_passed &= test_import("BaseTask", "dvisionix.training.tasks")
all_passed &= test_import("Callback", "dvisionix.training.callbacks.base")
all_passed &= test_import("Losses", "dvisionix.models.losses")

# 指标模块
print("\n指标模块:")
all_passed &= test_import("metrics.classification", "dvisionix.metrics.classification")
all_passed &= test_import("metrics.segmentation", "dvisionix.metrics.segmentation")
all_passed &= test_import("metrics.detection", "dvisionix.metrics.detection")
all_passed &= test_import("metrics.collection", "dvisionix.metrics.collection")
all_passed &= test_import("metrics.presets", "dvisionix.metrics.presets")

# 模型
print("\n模型:")
all_passed &= test_import("SimpleCNN", "dvisionix.models.base")
all_passed &= test_import("SimpleSegmentationModel", "dvisionix.models.base")
all_passed &= test_import("SequentialBackbone", "dvisionix.models.backbones")

print("\n" + "=" * 60)
if all_passed:
    print("[OK] 所有模块导入成功！")
else:
    print("[FAIL] 部分模块导入失败！")
print("=" * 60)
