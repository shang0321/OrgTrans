import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet_moco import WaveletTransform

class PseudoLabelGenerator(nn.Module):
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.5):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.wavelet = WaveletTransform()
        
        self.feature_bank = []
        self.label_bank = []
        self.max_bank_size = 1000
        
    def generate_pseudo_labels(self, teacher_pred, student_pred, features):
        LL, LH, HL, HH = self.wavelet(features)
        
        det_quality = self.compute_detection_quality(teacher_pred)
        
        feat_quality = self.compute_feature_quality(LL)
        
        edge_quality = self.compute_edge_quality(LH, HL, HH)
        
        quality_scores = (det_quality * 0.4 + 
                        feat_quality * 0.3 + 
                        edge_quality * 0.3)
        
        pseudo_labels = self.generate_labels(
            teacher_pred, quality_scores)
            
        self.update_feature_bank(LL, pseudo_labels)
        
        return pseudo_labels
        
    def compute_detection_quality(self, predictions):
        confidence = predictions['scores']
        
        class_prob = F.softmax(predictions['logits'], dim=-1)
        max_prob = torch.max(class_prob, dim=-1)[0]
        
        box_quality = self.compute_box_quality(predictions['boxes'])
        
        quality = confidence * max_prob * box_quality
        return quality
        
    def compute_feature_quality(self, features):
        features = F.normalize(features, dim=1)
        
        if len(self.feature_bank) > 0:
            bank = torch.stack(self.feature_bank)
            similarity = torch.mm(features, bank.T)
            quality = torch.max(similarity, dim=1)[0]
        else:
            quality = torch.ones_like(features[:, 0])
            
        return quality
        
    def compute_edge_quality(self, LH, HL, HH):
        h_edge = torch.mean(torch.abs(LH), dim=1)
        v_edge = torch.mean(torch.abs(HL), dim=1)
        d_edge = torch.mean(torch.abs(HH), dim=1)
        
        edge_quality = (h_edge + v_edge + d_edge) / 3
        return edge_quality
        
    def compute_box_quality(self, boxes):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        aspect_ratio = torch.min(w/h, h/w)
        
        area = w * h
        area_score = torch.clamp(area / (area.max() + 1e-6), 0, 1)
        
        return aspect_ratio * area_score
        
    def generate_labels(self, predictions, quality_scores):
        mask = quality_scores > self.confidence_threshold
        
        if not torch.any(mask):
            return None
            
        boxes = predictions['boxes'][mask]
        scores = predictions['scores'][mask]
        labels = predictions['labels'][mask]
        
        keep = self.nms(boxes, scores)
        
        pseudo_labels = {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep],
            'quality': quality_scores[mask][keep]
        }
        
        return pseudo_labels
        
    def nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        _, order = scores.sort(0, descending=True)
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i)

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= self.nms_threshold).nonzero().squeeze()
            
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return torch.LongTensor(keep)
        
    def update_feature_bank(self, features, pseudo_labels):
        if pseudo_labels is None:
            return
            
        for feat in features:
            self.feature_bank.append(feat)
            
        if len(self.feature_bank) > self.max_bank_size:
            self.feature_bank = self.feature_bank[-self.max_bank_size:]
            
class PseudoLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, student_pred, pseudo_labels):
        if pseudo_labels is None:
            return torch.tensor(0.0).cuda()
            
        cls_loss = F.cross_entropy(
            student_pred['logits'], 
            pseudo_labels['labels']
        )
        
        reg_loss = F.smooth_l1_loss(
            student_pred['boxes'],
            pseudo_labels['boxes']
        )
        
        loss = cls_loss + reg_loss
        return loss * pseudo_labels['quality'].mean() 