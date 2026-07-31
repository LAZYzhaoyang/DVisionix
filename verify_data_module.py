# -*- coding: utf-8 -*-
"""Data module quick smoke test.

Note: comprehensive coverage lives in ``tests/test_data/``. This script is just
a one-shot sanity check that the new unified data layer (Sample contract,
BaseDataset, atomic transforms, presets, build_dataset) can be imported and
instantiated end-to-end.
"""

import os
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _section(title: str) -> None:
    print()
    print(title)


def main() -> int:
    print("=" * 60)
    print("DVisionix data module smoke test")
    print("=" * 60)

    _section("1. Imports")
    from dvisionix.data import (
        BaseDataset,
        CustomDataset,
        ClassificationTransforms,
        DetectionTransforms,
        SegmentationTransforms,
        Sample,
        build_dataset,
    )
    from dvisionix.registry import DATASETS, TRANSFORMS
    print("   OK")

    _section("2. Registries")
    expected_datasets = {"base_dataset", "custom", "cifar10", "cifar100",
                         "imagenet", "imagefolder", "coco_detection",
                         "voc_detection", "cityscapes", "voc_segmentation",
                         "ade20k"}
    missing = expected_datasets - set(DATASETS.keys())
    assert not missing, f"Missing datasets: {missing}"
    print(f"   OK  ({len(DATASETS)} datasets registered)")

    expected_transforms = {"ImageResize", "BoxSyncResize", "ToTensor",
                            "ImageNormalize", "ClassificationTransforms",
                            "DetectionTransforms", "SegmentationTransforms",
                            "classification_transforms", "detection_transforms",
                            "segmentation_transforms", "albumentations"}
    missing_t = expected_transforms - set(TRANSFORMS.keys())
    assert not missing_t, f"Missing transforms: {missing_t}"
    print(f"   OK  ({len(TRANSFORMS)} transforms registered)")

    _section("3. CustomDataset (no transforms)")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmpdir, f"img_{i}.png"), img)

        samples = [
            {"image": os.path.join(tmpdir, f"img_{i}.png"), "label": i % 2}
            for i in range(3)
        ]
        ds = CustomDataset(samples=samples, task_type="classification")
        assert len(ds) == 3
        s = ds[0]
        assert isinstance(s, dict)
        assert s["image"].shape == (64, 64, 3)  # numpy HWC, no transforms applied
        assert int(s["label"]) == 0
    print("   OK")

    _section("4. CustomDataset + ClassificationTransforms")
    with tempfile.TemporaryDirectory() as tmpdir:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = os.path.join(tmpdir, "test.png")
        cv2.imwrite(img_path, img)
        ds = CustomDataset(
            samples=[{"image": img_path, "label": 0}],
            task_type="classification",
            transforms=ClassificationTransforms(train=True, image_size=32),
        )
        s = ds[0]
        # image should now be a tensor (3, 32, 32)
        assert hasattr(s["image"], "shape") and len(s["image"].shape) == 3
        assert s["image"].shape == (3, 32, 32)
    print("   OK")

    _section("5. build_dataset registry factory")
    cfg = {"type": "custom", "samples": [{"image": "dummy.png", "label": 0}]}
    ds = build_dataset(cfg)
    assert isinstance(ds, BaseDataset)
    print(f"   OK  -> {type(ds).__name__}")

    print()
    print("=" * 60)
    print("Data module smoke test passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())