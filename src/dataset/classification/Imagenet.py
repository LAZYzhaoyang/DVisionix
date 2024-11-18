from torchvision.datasets import imagenet
import os
import json
from PIL import Image
from typing import Dict, Any,Optional
from .cls_base import ClsDataset

'''
Implementation of the ImageNet dataset
The directory structure of the dataset is as follows
data_root/
    ├── train/
    │   ├── n01440764/
    │   ├── n01443537/
    │   └── ...
    ├── val/
    │   ├── n01440764/
    │   ├── n01443537/
    │   └── ...
    └── imagenet_class_index.json
Where:
imagenet_class_index.json contains the mapping from class ID to name
Each category folder (e.g., n01440764) contains all images of that category
Supports two modes: train and val
The returned data includes image, label, and image path information
'''

class ImageNetDataset(ClsDataset):
    """ImageNet Dataset Implementation"""
    
    def _init_dataset(self) -> None:
        """Initialize ImageNet dataset structure"""
        # Load ImageNet class mapping
        mapping_file = os.path.join(self.data_root, 'imagenet_class_index.json')
        with open(mapping_file, 'r') as f:
            class_mapping = json.load(f)
            
        # Setup category information
        for _, (class_id, class_name) in class_mapping.items():
            self.categories.append(class_name)
            self.class_to_idx[class_id] = len(self.categories) - 1
            
        # Get image paths based on mode (train/val)
        data_dir = os.path.join(self.data_root, self.mode)
        for class_id in self.class_to_idx.keys():
            class_dir = os.path.join(data_dir, class_id)
            if not os.path.exists(class_dir):
                continue
                
            # Collect all images for this class
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                self.data_infos.append({
                    'img_path': os.path.join(class_dir, img_name),
                    'label': self.class_to_idx[class_id]
                })
    
    def _load_data(self, index: int) -> Dict[str, Any]:
        """
        Load a single image and its label
        
        Returns:
            Dict containing:
                'img': PIL Image
                'label': class index
                'img_path': path to image file
        """
        data_info = self.data_infos[index]
        
        # Load image
        img = Image.open(data_info['img_path']).convert('RGB')
        
        return {
            'img': img,
            'label': data_info['label'],
            'img_path': data_info['img_path']
        }
    
    def evaluate(self, 
                results: list, 
                metrics: Optional[list] = None, 
                **kwargs) -> Dict[str, float]:
        """
        Evaluate prediction results
        
        Args:
            results: List of predicted labels
            metrics: List of metrics to compute (default: ['accuracy'])
        """
        if metrics is None:
            metrics = ['accuracy']
            
        eval_results = {}
        
        # Calculate accuracy
        if 'accuracy' in metrics:
            correct = sum(pred == self.data_infos[i]['label'] 
                         for i, pred in enumerate(results))
            eval_results['accuracy'] = correct / len(results)
            
        return eval_results