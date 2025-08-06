import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, data_file, img_size=640, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        
        with open(data_file, 'r') as f:
            self.img_files = [line.strip() for line in f.readlines()]
            
        self.label_files = []
        for img_file in self.img_files:
            label_file = img_file.replace('images', 'labels').replace('.jpg', '.txt')
            self.label_files.append(label_file)
            
        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        label_path = self.label_files[idx]
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path)
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
        else:
            labels = np.zeros((0, 5))
            
        boxes = labels[:, 1:] if len(labels) > 0 else np.zeros((0, 4))
        class_labels = labels[:, 0] if len(labels) > 0 else np.zeros(0)
        
        transformed = self.transform(
            image=img,
            bboxes=boxes,
            class_labels=class_labels
        )
        
        img = torch.from_numpy(transformed['image'].transpose(2, 0, 1))
        boxes = torch.from_numpy(np.array(transformed['bboxes']))
        class_labels = torch.from_numpy(np.array(transformed['class_labels']))
        
        return {
            'image': img,
            'boxes': boxes,
            'labels': class_labels.long(),
            'img_path': img_path
        } 