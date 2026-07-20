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
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
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
all_passed &= test_import("ClassificationDatasetAdapter", "dvisionix.data.adapters.classification")
all_passed &= test_import("DetectionDatasetAdapter", "dvisionix.data.adapters.detection")
all_passed &= test_import("SegmentationDatasetAdapter", "dvisionix.data.adapters.segmentation")

# 数据变换
print("\n数据变换:")
all_passed &= test_import("ClassificationTransforms", "dvisionix.data.transforms.classification")
all_passed &= test_import("DetectionTransforms", "dvisionix.data.transforms.detection")
all_passed &= test_import("SegmentationTransforms", "dvisionix.data.transforms.segmentation")

# 训练模块
print("\n训练模块:")
all_passed &= test_import("Trainer", "dvisionix.training.trainer")
all_passed &= test_import("BaseTask", "dvisionix.training.task")
all_passed &= test_import("Callback", "dvisionix.training.callbacks")
all_passed &= test_import("Losses", "dvisionix.training.losses")

# 指标模块
print("\n指标模块:")
all_passed &= test_import("ClassificationMetrics", "dvisionix.metrics.classification")
all_passed &= test_import("SegmentationMetrics", "dvisionix.metrics.segmentation")
all_passed &= test_import("DetectionMetrics", "dvisionix.metrics.detection")

# 模型
print("\n模型:")
all_passed &= test_import("SimpleCNN", "dvisionix.models.base")
all_passed &= test_import("SimpleSegmentationModel", "dvisionix.models.base")
all_passed &= test_import("SimpleDetectionModel", "dvisionix.models.base")

print("\n" + "=" * 60)
if all_passed:
    print("✓ 所有模块导入成功！")
else:
    print("✗ 部分模块导入失败！")
print("=" * 60)
