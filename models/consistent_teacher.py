import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ConsistentTeacher(nn.Module):
    def __init__(self, num_classes=80, pretrained=True):
        super().__init__()
        self.student = resnet50(pretrained=pretrained)
        self.teacher = resnet50(pretrained=pretrained)
        
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        in_features = self.student.fc.in_features
        self.student.fc = nn.Identity()
        self.teacher.fc = nn.Identity()
        
        self.student_det_head = DetectionHead(in_features, num_classes)
        self.teacher_det_head = DetectionHead(in_features, num_classes)
        
        self.momentum = 0.999
        
    def forward(self, images, targets=None, is_train=True):
        if is_train:
            return self.forward_train(images, targets)
        else:
            return self.forward_test(images)
            
    def forward_train(self, images, targets):
        student_feat = self.student(images)
        student_pred = self.student_det_head(student_feat)
        
        with torch.no_grad():
            teacher_feat = self.teacher(images)
            teacher_pred = self.teacher_det_head(teacher_feat)
            
        det_loss = self.student_det_head.loss(student_pred, targets)
        consist_loss = self.consistency_loss(student_pred, teacher_pred)
        
        losses = {
            'det_loss': det_loss,
            'consist_loss': consist_loss,
            'total_loss': det_loss + consist_loss
        }
        
        return losses
        
    def forward_test(self, images):
        with torch.no_grad():
            feat = self.teacher(images)
            pred = self.teacher_det_head(feat)
        return pred
        
    def consistency_loss(self, student_pred, teacher_pred):
        loss = F.mse_loss(student_pred['bbox_pred'], teacher_pred['bbox_pred']) + \
               F.mse_loss(student_pred['cls_pred'], teacher_pred['cls_pred'])
        return loss
        
    @torch.no_grad()
    def update_teacher(self):
        for student_param, teacher_param in zip(self.student.parameters(), 
                                              self.teacher.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + \
                                (1 - self.momentum) * student_param.data
                                
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 1)
        )
        
    def forward(self, x):
        feat = self.conv(x)
        return {
            'cls_pred': self.cls_head(feat),
            'bbox_pred': self.bbox_head(feat)
        }
        
    def loss(self, pred, targets):
        cls_loss = F.cross_entropy(pred['cls_pred'], targets['labels'])
        bbox_loss = F.smooth_l1_loss(pred['bbox_pred'], targets['boxes'])
        return cls_loss + bbox_loss 