# -*- coding: utf-8 -*-
"""主流公开数据集预设。

目的：让用户 ``build_dataset({"type": "cifar10", ...})`` 就能开箱用主流数据集，
不用自己写样本列表或 Adapter。具体的加载/解析逻辑都封装在这里。

约定：
- 每个数据集类都 ``@DATASETS.register()``，注册名用数据集通用简称
  （如 ``cifar10`` / ``coco_detection`` / ``cityscapes``），可同时挂多个别名。
- 子类需要实现 ``_build_samples(root, train, **kwargs) -> List[dict]``，
  返回符合 ``Sample`` 契约的样本列表（每个 dict 必含 ``image`` 字段）。
- 训练 / 验证样本列表用 ``train`` 参数区分。
"""

from typing import Any, Dict, List

import numpy as np

from ..registry import DATASETS
from .base import BaseDataset
from .collate import detection_collate, segmentation_collate


class _ImageFolderDataset(BaseDataset):
    """ImageFolder 风格数据集：``root/{class_name}/*.jpg``。"""

    task_type = "classification"

    def __init__(
        self, root: str, train: bool = True, split: str = "train", transforms=None, **kwargs
    ):
        import os

        split_dir = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"ImageFolder root not found: {split_dir}")
        classes = sorted(
            d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
        )
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        samples = []
        for c in classes:
            for fname in os.listdir(os.path.join(split_dir, c)):
                p = os.path.join(split_dir, c, fname)
                if os.path.isfile(p):
                    samples.append({"image": p, "label": cls_to_idx[c]})
        super().__init__(
            samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )
        self.classes = classes


@DATASETS.register()
@DATASETS.register(name="imagefolder")
class ImageFolderDataset(_ImageFolderDataset):
    """ImageFolder 分类数据集（``root/{class_name}/*.jpg``）。"""


@DATASETS.register()
@DATASETS.register(name="cifar10")
class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 分类数据集（需 ``torchvision``，自动下载）。"""

    task_type = "classification"

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transforms=None,
        **kwargs,
    ):
        from torchvision.datasets import CIFAR10

        ds = CIFAR10(root=root, train=train, download=download)
        # CIFAR-10 数据在内存中，转成缓存文件以便 transforms 走统一图像加载路径
        import os

        cache_dir = os.path.join(root, "cifar10_cache", "train" if train else "val")
        os.makedirs(cache_dir, exist_ok=True)
        import cv2

        samples = []
        for idx in range(len(ds)):
            img_pil, label = ds[idx]
            img_path = os.path.join(cache_dir, f"{idx:06d}.png")
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR))
            samples.append({"image": img_path, "label": int(label)})
        super().__init__(
            samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )
        self.classes = ds.classes


@DATASETS.register()
@DATASETS.register(name="cifar100")
class CIFAR100Dataset(CIFAR10Dataset):
    """CIFAR-100 分类数据集。"""

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transforms=None,
        **kwargs,
    ):
        from torchvision.datasets import CIFAR100

        ds = CIFAR100(root=root, train=train, download=download)
        import os

        import cv2

        cache_dir = os.path.join(root, "cifar100_cache", "train" if train else "val")
        os.makedirs(cache_dir, exist_ok=True)
        samples = []
        for idx in range(len(ds)):
            img_pil, label = ds[idx]
            img_path = os.path.join(cache_dir, f"{idx:06d}.png")
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR))
            samples.append({"image": img_path, "label": int(label)})
        BaseDataset.__init__(
            self, samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )
        self.classes = ds.classes


@DATASETS.register()
@DATASETS.register(name="imagenet")
class ImageNetDataset(_ImageFolderDataset):
    """ImageNet 分类数据集（用 ImageFolder 风格 ``root/{train,val}/{wnid}/*.JPEG``）。"""

    def __init__(self, root: str, train: bool = True, transforms=None, **kwargs):
        super().__init__(
            root, train=train, split="train" if train else "val", transforms=transforms, **kwargs
        )


class _CocoBaseDataset(BaseDataset):
    """COCO 数据集公共基类。"""

    task_type = "detection"
    collate_fn = staticmethod(detection_collate)

    def _build_samples(self, annotation_file: str) -> List[Dict[str, Any]]:
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pycocotools is required for COCO datasets. pip install pycocotools"
            ) from exc
        import os

        coco = COCO(annotation_file)
        img_dir = os.path.dirname(annotation_file).rstrip("/")
        if img_dir.endswith("annotations"):
            img_dir = os.path.dirname(img_dir)
        samples = []
        for img_id in coco.imgs:
            img_info = coco.imgs[img_id]
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            boxes, labels = [], []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(int(a["category_id"]))
            samples.append(
                {
                    "image": os.path.join(img_dir, img_info["file_name"]),
                    "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "labels": np.asarray(labels, dtype=np.int64),
                }
            )
        return samples


@DATASETS.register()
@DATASETS.register(name="coco_detection")
@DATASETS.register(name="coco")
class CocoDetectionDataset(_CocoBaseDataset):
    """COCO 检测数据集（需 pycocotools）。

    Args:
        root: 包含 ``annotations/instances_*.json`` 的目录。
        split: ``train`` / ``val``（决定加载哪个 annotation 文件）。
    """

    def __init__(
        self, root: str, split: str = "train", year: str = "2017", transforms=None, **kwargs
    ):
        ann_file = f"{root}/annotations/instances_{split}{year}.json"
        samples = self._build_samples(ann_file)
        super().__init__(
            samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )


@DATASETS.register()
@DATASETS.register(name="voc_detection")
class VOCDetectionDataset(BaseDataset):
    """Pascal VOC 检测数据集（torchvision.datasets.VOCDetection 包装）。"""

    task_type = "detection"
    collate_fn = staticmethod(detection_collate)

    def __init__(
        self, root: str, year: str = "2012", image_set: str = "train", transforms=None, **kwargs
    ):
        from torchvision.datasets import VOCDetection

        ds = VOCDetection(
            root=root, year=year, image_set=image_set, download=kwargs.get("download", False)
        )
        import numpy as np

        VOC_LABELS = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        cls_to_idx = {c: i for i, c in enumerate(VOC_LABELS)}
        samples = []
        for img, target in ds:
            boxes, labels = [], []
            for obj in target["annotation"].get("object", []):
                diff = int(obj.get("difficult", 0))
                if diff == 1:
                    continue
                bb = obj["bndbox"]
                x1, y1 = float(bb["xmin"]) - 1, float(bb["ymin"]) - 1
                x2, y2 = float(bb["xmax"]) - 1, float(bb["ymax"]) - 1
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_to_idx[obj["name"]])
            samples.append(
                {
                    "image": img.filename if hasattr(img, "filename") else str(img),
                    "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "labels": np.asarray(labels, dtype=np.int64),
                }
            )
        super().__init__(
            samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )
        self.classes = VOC_LABELS


class _CityscapesBaseDataset(BaseDataset):
    """Cityscapes 分割数据集。"""

    task_type = "segmentation"
    collate_fn = staticmethod(segmentation_collate)

    def _build_samples(self, root: str, split: str) -> List[Dict[str, Any]]:
        import os

        img_dir = os.path.join(root, "leftImg8bit", split)
        gt_dir = os.path.join(root, "gtFine", split)
        samples = []
        for city in sorted(os.listdir(img_dir)):
            city_img_dir = os.path.join(img_dir, city)
            city_gt_dir = os.path.join(gt_dir, city)
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                img_path = os.path.join(city_img_dir, fname)
                mask_name = fname.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                mask_path = os.path.join(city_gt_dir, mask_name)
                if os.path.isfile(mask_path):
                    samples.append({"image": img_path, "mask": mask_path})
        return samples


@DATASETS.register()
@DATASETS.register(name="cityscapes")
class CityscapesDataset(_CityscapesBaseDataset):
    def __init__(self, root: str, split: str = "train", transforms=None, **kwargs):
        super().__init__(
            self._build_samples(root, split),
            transforms=transforms,
            return_meta=kwargs.get("return_meta", False),
        )


@DATASETS.register()
@DATASETS.register(name="ade20k")
class ADE20KDataset(BaseDataset):
    """ADE20K 分割数据集（ImageFolder 风格 ``images/{split}/ADE_train_*.jpg`` + ``annotations/{split}/ADE_train_*.png``）。"""

    task_type = "segmentation"
    collate_fn = staticmethod(segmentation_collate)

    def __init__(self, root: str, split: str = "train", transforms=None, **kwargs):
        import os

        img_dir = os.path.join(root, "images", split)
        mask_dir = os.path.join(root, "annotations", split)
        samples = []
        for fname in sorted(os.listdir(img_dir)):
            mask_name = fname.replace(".jpg", ".png")
            img_path, mask_path = os.path.join(img_dir, fname), os.path.join(mask_dir, mask_name)
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                samples.append({"image": img_path, "mask": mask_path})
        super().__init__(
            samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )


@DATASETS.register()
@DATASETS.register(name="voc_segmentation")
class VOCSegmentationDataset(BaseDataset):
    """Pascal VOC 分割数据集。"""

    task_type = "segmentation"
    collate_fn = staticmethod(segmentation_collate)

    def __init__(
        self, root: str, year: str = "2012", image_set: str = "train", transforms=None, **kwargs
    ):
        from torchvision.datasets import VOCSegmentation

        ds = VOCSegmentation(
            root=root, year=year, image_set=image_set, download=kwargs.get("download", False)
        )
        samples = [
            {
                "image": img.filename if hasattr(img, "filename") else str(img),
                "mask": mask.filename if hasattr(mask, "filename") else str(mask),
            }
            for img, mask in ds
        ]
        super().__init__(
            samples, transforms=transforms, return_meta=kwargs.get("return_meta", False)
        )


__all__ = [
    "ImageFolderDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "ImageNetDataset",
    "CocoDetectionDataset",
    "VOCDetectionDataset",
    "CityscapesDataset",
    "VOCSegmentationDataset",
    "ADE20KDataset",
]
