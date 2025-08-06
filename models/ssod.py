import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .moco import MoCo, MoCoLoss, moco_augmentation

class SSOD:
    def __init__(self, model, hyp, ssod_hyp):
        self.model = model
        self.hyp = hyp
        self.ssod_hyp = ssod_hyp
        
        self.moco = MoCo(base_encoder=model.backbone, 
                        dim=512,
                        K=65536,
                        m=0.999,
                        T=0.07)
        
        self.moco_loss = MoCoLoss()
        
        self.ema_model = ModelEMA(model)
        self.ema_threshold = 0.5
        self.min_threshold = 0.01
        self.max_threshold = 0.7
        self.ema_alpha = 0.99
        
        self.quality_history = []
        self.max_history = 100
        self.pseudo_label_buffer = []
        self.buffer_size = 1000
        self.consistency_weight = 0.5

    def forward_train(self, imgs, targets=None):
        im_q, im_k = moco_augmentation(imgs)
        moco_logits, moco_labels = self.moco(im_q, im_k)
        moco_loss = self.moco_loss(moco_logits, moco_labels)
        
        det_loss = self.model(imgs, targets)
        
        with torch.no_grad():
            pseudo_labels = self.generate_pseudo_labels(imgs)
        
        if pseudo_labels:
            pseudo_loss = self.model(imgs, pseudo_labels)
        else:
            pseudo_loss = torch.tensor(0.0).to(imgs.device)
        
        total_loss = det_loss + self.ssod_hyp.get('moco_weight', 0.1) * moco_loss + \
                    self.ssod_hyp.get('pseudo_weight', 0.3) * pseudo_loss
                    
        return {
            'total_loss': total_loss,
            'det_loss': det_loss,
            'moco_loss': moco_loss,
            'pseudo_loss': pseudo_loss
        }

    def generate_pseudo_labels(self, imgs):
        device = imgs.device
        with torch.no_grad():
            feat_k = self.moco.encoder_k(imgs)
            feat_k = F.normalize(feat_k, dim=1)
            
            sim_matrix = torch.mm(feat_k, self.moco.queue.clone().detach())
            sim_scores = torch.max(sim_matrix, dim=1)[0]
            
            preds = self.ema_model(imgs)
            
            quality = self.compute_prediction_quality(preds)
            self.quality_history.append(quality)
            if len(self.quality_history) > self.max_history:
                self.quality_history.pop(0)
            
            avg_quality = sum(self.quality_history) / len(self.quality_history)
            self.ema_threshold = self.min_threshold + \
                               (self.max_threshold - self.min_threshold) * avg_quality
            
            pseudo_labels = []
            for i, (pred, sim_score) in enumerate(zip(preds, sim_scores)):
                pred = non_max_suppression(pred, 
                                         conf_thres=self.ema_threshold,
                                         iou_thres=self.ssod_hyp.get('nms_iou_thres', 0.65))
                
                for j, det in enumerate(pred):
                    if len(det):
                        quality_scores = self.compute_box_quality(det)
                        final_scores = quality_scores * 0.7 + sim_score * 0.3
                        mask = final_scores > self.ema_threshold
                        det = det[mask]
                        
                        if len(det):
                            consistency_scores = self.compute_consistency(det)
                            final_scores = final_scores[mask] * (1 - self.consistency_weight) + \
                                         consistency_scores * self.consistency_weight
                            mask = final_scores > self.ema_threshold
                            det = det[mask]
                            
                            if len(det):
                                pseudo_labels.append({
                                    'boxes': det[:, :4],
                                    'scores': det[:, 4],
                                    'labels': det[:, 5]
                                })
            
            self.pseudo_label_buffer.extend(pseudo_labels)
            if len(self.pseudo_label_buffer) > self.buffer_size:
                self.pseudo_label_buffer = self.pseudo_label_buffer[-self.buffer_size:]
            
            return pseudo_labels

    def compute_prediction_quality(self, preds):
        quality = 0.0
        for pred in preds:
            cls_conf = torch.sigmoid(pred[..., 4:]).max(dim=-1)[0]
            obj_conf = torch.sigmoid(pred[..., 4])
            box_stability = 1.0 - torch.var(pred[..., :4], dim=-1).mean()
            quality += (cls_conf * obj_conf * box_stability).mean().item()
        return quality / len(preds)

    def compute_box_quality(self, det):
        cls_conf = det[:, 4]
        obj_conf = det[:, 4]
        box_stability = 1.0 - torch.var(det[:, :4], dim=0).mean()
        return cls_conf * obj_conf * box_stability

    def compute_consistency(self, det):
        if len(self.pseudo_label_buffer) == 0:
            return torch.ones(len(det), device=det.device)
        
        consistency_scores = []
        for box in det[:, :4]:
            max_iou = 0.0
            for hist_label in self.pseudo_label_buffer:
                hist_boxes = hist_label['boxes']
                ious = self.compute_iou(box, hist_boxes)
                max_iou = max(max_iou, ious.max().item())
            consistency_scores.append(max_iou)
        return torch.tensor(consistency_scores, device=det.device)

    def compute_iou(self, box, boxes):
        box = box.unsqueeze(0)
        x1 = torch.max(box[:, 0], boxes[:, 0])
        y1 = torch.max(box[:, 1], boxes[:, 1])
        x2 = torch.min(box[:, 2], boxes[:, 2])
        y2 = torch.min(box[:, 3], boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        return intersection / (union + 1e-16) 