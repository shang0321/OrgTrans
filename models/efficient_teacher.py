import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet_moco import WaveletMoCo, WaveletMoCoLoss
from .pseudo_label import PseudoLabelGenerator, PseudoLabelLoss

class EfficientTeacher(nn.Module):
    def __init__(self, base_model, num_classes, 
                 moco_dim=128, moco_k=65536, moco_m=0.999, moco_t=0.07):
        super().__init__()
        self.student_model = base_model
        self.teacher_model = base_model.__class__(**base_model.config)
        
        self.moco = WaveletMoCo(
            base_encoder=base_model.backbone,
            dim=moco_dim,
            K=moco_k,
            m=moco_m,
            T=moco_t
        )
        
        self.pseudo_label_generator = PseudoLabelGenerator()
        
        self.moco_loss = WaveletMoCoLoss()
        self.pseudo_label_loss = PseudoLabelLoss()
        
        self._init_teacher()
        
    def _init_teacher(self):
        for param_t, param_s in zip(self.teacher_model.parameters(),
                                  self.student_model.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
            
    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_t, param_s in zip(self.teacher_model.parameters(),
                                  self.student_model.parameters()):
            param_t.data = param_t.data * self.moco.m + \
                          param_s.data * (1. - self.moco.m)
                          
    def forward_train(self, labeled_data, unlabeled_data):
        losses = {}
        
        if labeled_data is not None:
            sup_loss = self.forward_supervised(labeled_data)
            losses.update(sup_loss)
        
        if unlabeled_data is not None:
            unsup_loss = self.forward_unsupervised(unlabeled_data)
            losses.update(unsup_loss)
            
        return losses
        
    def forward_supervised(self, data):
        images, targets = data
        
        student_pred = self.student_model(images)
        
        det_loss = self.student_model.loss(student_pred, targets)
        
        return {'det_loss': det_loss}
        
    def forward_unsupervised(self, data):
        weak_aug, strong_aug = data
        
        moco_out = self.moco(weak_aug, strong_aug)
        moco_loss = self.moco_loss(*moco_out)
        
        with torch.no_grad():
            teacher_pred = self.teacher_model(weak_aug)
            
        pseudo_labels = self.pseudo_label_generator(
            teacher_pred,
            None,
            self.moco.encoder_k(weak_aug)
        )
        
        student_pred = self.student_model(strong_aug)
        
        if pseudo_labels is not None:
            pseudo_loss = self.pseudo_label_loss(
                student_pred, pseudo_labels)
        else:
            pseudo_loss = torch.tensor(0.0).cuda()
            
        self._momentum_update_teacher()
        
        return {
            'moco_loss': moco_loss,
            'pseudo_loss': pseudo_loss
        }
        
    def forward_test(self, data):
        return self.teacher_model(data)
        
    def forward(self, data, mode='train'):
        if mode == 'train':
            return self.forward_train(*data)
        else:
            return self.forward_test(data) 