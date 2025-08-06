import torch
import torch.nn as nn
import torchvision.transforms as T
import random
import numpy as np

class RandomResizedCrop(nn.Module):
    def __init__(self, size=224, scale=(0.2, 1.0)):
        super().__init__()
        self.size = size
        self.scale = scale
        
    def forward(self, img):
        return T.RandomResizedCrop(self.size, scale=self.scale)(img)

class RandomColorJitter(nn.Module):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        super().__init__()
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
    def forward(self, img):
        return self.transform(img)

class RandomGrayscale(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.transform = T.RandomGrayscale(p=p)
        
    def forward(self, img):
        return self.transform(img)

class RandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size=23, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def forward(self, img):
        if random.random() < 0.5:
            return T.GaussianBlur(self.kernel_size, sigma=self.sigma)(img)
        return img

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.transform = T.RandomHorizontalFlip(p=p)
        
    def forward(self, img):
        return self.transform(img)

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.transform = T.Normalize(mean=mean, std=std)
        
    def forward(self, img):
        return self.transform(img)

class MoCoTransform:
    def __init__(self, size=224):
        self.transform = nn.Sequential(
            RandomResizedCrop(size),
            RandomColorJitter(),
            RandomGrayscale(),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
            Normalize()
        )
    
    def __call__(self, img):
        img_q = self.transform(img)
        img_k = self.transform(img)
        return img_q, img_k

def build_moco_transform(img_size=224):
    return MoCoTransform(size=img_size) 