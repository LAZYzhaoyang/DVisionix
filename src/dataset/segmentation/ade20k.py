import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
from .seg_base import SegDataset

'''
Implementation of the ADE20K dataset
The directory structure of the dataset is as follows
data_root/
    ├── images/
    │   ├── training/
    │   │   ├── ADE_train_00000001.jpg
    │   │   └── ...
    │   └── validation/
    │       ├── ADE_val_00000001.jpg
    │       └── ...
    └── annotations/
        ├── training/
        │   ├── ADE_train_00000001.png
        │   └── ...
        └── validation/
            ├── ADE_val_00000001.png
            └── ...
'''

class ADE20KDataset(SegDataset):
    """ADE20K Dataset Implementation"""

    # ADE20K default categories
    ADE20K_CATEGORIES = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'] # ... add all 150 categories
    
    # ADE20K color map (RGB values for visualization)
    ADE20K_COLORS = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize categories
        self.categories = self.ADE20K_CATEGORIES
        # Initialize color map
        self.label_map = {i: color for i, color in enumerate(self.ADE20K_COLORS)}
        
    def _init_dataset(self) -> None:
        """Initialize ADE20K dataset structure"""
        # Set up paths based on mode
        if self.mode == 'train':
            img_dir = os.path.join(self.data_root, 'images/training')
            mask_dir = os.path.join(self.data_root, 'annotations/training')
        else:
            img_dir = os.path.join(self.data_root, 'images/validation')
            mask_dir = os.path.join(self.data_root, 'annotations/validation')
            
        # Collect all image and mask pairs
        for img_name in os.listdir(img_dir):
            if not img_name.endswith('.jpg'):
                continue
                
            mask_name = img_name.replace('.jpg', '.png')
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            # Verify mask file exists
            if not os.path.exists(mask_path):
                continue
                
            self.data_infos.append({
                'img_path': img_path,
                'mask_path': mask_path,
                'filename': img_name
            })
    
    def _load_data(self, index: int) -> Dict[str, Any]:
        """
        Load a single image and its segmentation mask
        
        Returns:
            Dict containing:
                'img': PIL Image
                'mask': numpy array of shape (H, W)
                'filename': image filename
        """
        info = self.data_infos[index]
        
        # Load image
        img = Image.open(info['img_path']).convert('RGB')
        
        # Load mask
        mask = Image.open(info['mask_path'])
        mask = np.array(mask, dtype=np.int64)
        
        # ADE20K masks need to subtract 1 as labels begin from 1
        mask = mask - 1
        mask[mask == -1] = self.ignore_index
        
        return {
            'img': img,
            'mask': mask,
            'filename': info['filename']
        }
    
    def encode_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to training format"""
        # ADE20K specific mask processing if needed
        return mask
    
    def decode_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to visualization format with color mapping"""
        # Create RGB mask
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label, color in self.label_map.items():
            rgb_mask[mask == label] = color
            
        return rgb_mask
    
    def evaluate(self,
                results: list,
                metrics: Optional[list] = None,
                **kwargs) -> Dict[str, float]:
        """
        Evaluate prediction results
        
        Args:
            results: List of predicted segmentation masks
            metrics: List of metrics to compute (default: ['mIoU'])
        """
        if metrics is None:
            metrics = ['mIoU']
            
        eval_results = {}
        
        if 'mIoU' in metrics:
            # Calculate mean IoU
            total_iou = 0
            valid_classes = 0
            
            for pred, info in zip(results, self.data_infos):
                gt_mask = np.array(Image.open(info['mask_path'])) - 1
                for class_id in range(len(self.categories)):
                    pred_mask = (pred == class_id)
                    gt_mask_binary = (gt_mask == class_id)
                    
                    intersection = np.logical_and(pred_mask, gt_mask_binary).sum()
                    union = np.logical_or(pred_mask, gt_mask_binary).sum()
                    
                    if union > 0:
                        total_iou += intersection / union
                        valid_classes += 1
            
            eval_results['mIoU'] = total_iou / valid_classes if valid_classes > 0 else 0
            
        return eval_results