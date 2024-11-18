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
    ADE20K_CATEGORIES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 
        'window', 'grass', # ... add all 150 categories
    ]
    
    # ADE20K color map (RGB values for visualization)
    ADE20K_COLORS = [
        (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
        (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
        (230, 230, 230), (4, 250, 7), # ... add all 150 colors
    ]

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