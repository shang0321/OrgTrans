
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
import math
from torch.nn import functional as F
from torch.autograd import Variable,Function
import numpy as np
from utils.general import xywh2xyxy, box_iou
from assigner import YOLOAnchorAssigner

def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps

class ComputeStudentMatchLoss():
    def __init__(self, model, cfg):
        super(ComputeStudentMatchLoss, self).__init__()
        device = next(model.parameters()).device
        autobalance = cfg.Loss.autobalance
        cls_pw = cfg.Loss.cls_pw
        obj_pw = cfg.Loss.obj_pw
        label_smoothing = cfg.Loss.label_smoothing

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))

        if cfg.SSOD.focal_loss > 0:
            BCEobj = FocalLoss(BCEobj)

        self.cp, self.cn = smooth_BCE(eps=label_smoothing)

        det = model.module.head if is_parallel(model) else model.head
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr,  self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.box_w = cfg.SSOD.box_loss_weight
        self.obj_w = cfg.SSOD.obj_loss_weight
        self.cls_w = cfg.SSOD.cls_loss_weight * cfg.Dataset.nc / 80. * 3. / det.nl
        self.anchor_t = cfg.Loss.anchor_t
        self.ignore_thres_high = [cfg.SSOD.ignore_thres_high] * cfg.Dataset.nc
        self.ignore_thres_low = [cfg.SSOD.ignore_thres_low] * cfg.Dataset.nc
        self.uncertain_aug = cfg.SSOD.uncertain_aug
        self.use_ota = cfg.SSOD.use_ota
        self.ignore_obj = cfg.SSOD.ignore_obj
        self.pseudo_label_with_obj = cfg.SSOD.pseudo_label_with_obj
        self.pseudo_label_with_bbox = cfg.SSOD.pseudo_label_with_bbox
        self.pseudo_label_with_cls = cfg.SSOD.pseudo_label_with_cls
        self.num_keypoints = cfg.Dataset.np
        self.single_targets = False
        if not self.uncertain_aug:
            self.single_targets = True

        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))
        self.assigner = YOLOAnchorAssigner(self.na, self.nl, self.anchors, self.anchor_t, det.stride, \
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=self.use_ota)

    def select_targets(self, targets):
        device = targets.device
        reliable_targets = []
        uncertain_targets = []
        uncertain_obj_targets = []
        uncertain_cls_targets = []
        for t in targets:
            t = np.array(t.cpu())
            if t[6] >= self.ignore_thres_high[int(t[1])]:
                    reliable_targets.append(t[:7])
            elif t[6] >= self.ignore_thres_low[int(t[1])]:
                if self.pseudo_label_with_obj:
                    uncertain_targets.append(np.concatenate((t[:6], t[7:8])))
                    if t[7] >= 0.99:
                        uncertain_obj_targets.append(np.concatenate((t[:6], t[7:8])))
                    if t[8] >= 0.99:
                        uncertain_cls_targets.append(np.concatenate((t[:6], t[7:8])))
                else:
                    uncertain_targets.append(t[:7])

        reliable_targets = np.array(reliable_targets).astype(np.float32)
        reliable_targets= torch.from_numpy(reliable_targets).contiguous()
        reliable_targets = reliable_targets.to(device)
        if reliable_targets.shape[0] == 0:
            reliable_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_targets = np.array(uncertain_targets).astype(np.float32)
        uncertain_targets= torch.from_numpy(uncertain_targets).contiguous()
        uncertain_targets= uncertain_targets.to(device)
        if uncertain_targets.shape[0] == 0:
            uncertain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_obj_targets = np.array(uncertain_obj_targets).astype(np.float32)
        uncertain_obj_targets= torch.from_numpy(uncertain_obj_targets).contiguous()
        uncertain_obj_targets= uncertain_obj_targets.to(device)
        if uncertain_obj_targets.shape[0] == 0:
            uncertain_obj_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_cls_targets = np.array(uncertain_cls_targets).astype(np.float32)
        uncertain_cls_targets= torch.from_numpy(uncertain_cls_targets).contiguous()
        uncertain_cls_targets= uncertain_cls_targets.to(device)
        if uncertain_cls_targets.shape[0] == 0:
            uncertain_cls_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)
        return reliable_targets, uncertain_targets, uncertain_obj_targets, uncertain_cls_targets

    def default_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)        
        if targets.shape[1] > 6:
            certain_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            if self.uncertain_aug:
                tcls, tbox, indices, anchors  = self.assigner(p, certain_targets)
                _, _, uc_indices, _, uc_scores = self.assigner(p, uc_targets, with_pseudo_score=True)
                _, uc_tbox, uc_obj_indices, uc_anchors, _ = self.assigner(p, uc_obj_targets, with_pseudo_score=True)
                uc_tcls, _, uc_cls_indices, _, _ = self.assigner(p, uc_cls_targets, with_pseudo_score=True)
            else:
                tcls, tbox, indices, anchors  = self.assigner(p, certain_targets)
                _, _, uc_indices, _, uc_scores = self.assigner(p, uc_targets, with_pseudo_score=True)
                _, uc_tbox, uc_obj_indices, uc_anchors, _ = self.assigner(p, uc_obj_targets, with_pseudo_score=True)                
                uc_tcls, _, uc_cls_indices, _, _ = self.assigner(p, uc_cls_targets, with_pseudo_score=True)
        else:
            tcls, tbox, indices, anchors  = self.assigner(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            anchors_spec = anchors[i] 
            tbox_spec = tbox[i]
            n = b.shape[0]

            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox_spec, x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            if targets.shape[1] > 6:
                uc_b, uc_a, uc_gj, uc_gi = uc_indices[i]
                n = uc_b.shape[0]
                if n:
                    if self.ignore_obj:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = -1
                    else:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = uc_scores[i].type(tobj.dtype)

                if self.pseudo_label_with_bbox:
                    uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi = uc_obj_indices[i]
                    n = uc_obj_b.shape[0]
                    uc_tbox_spec = uc_tbox[i]
                    anchors_spec = uc_anchors[i]
                    if n:
                        uc_ps = pi[uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi]
                        pxy = uc_ps[:, :2].sigmoid() * 2. - 0.5
                        pwh = (uc_ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                        pbox = torch.cat((pxy, pwh), 1)
                        iou = bbox_iou(pbox.T, uc_tbox_spec, x1y1x2y2=False, CIoU=True)
                        lbox += (1.0 - iou).mean()
                
                if self.pseudo_label_with_cls:
                    uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi = uc_cls_indices[i]
                    n = uc_cls_b.shape[0]
                    if n:
                        uc_ps = pi[uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi]
                        if self.nc > 1:
                            t = torch.full_like(uc_ps[:, 5:], self.cn, device=device)
                            t[range(n), uc_tcls[i]] = self.cp
                            lcls += self.BCEcls(uc_ps[:, 5:], t)
            valid_mask = tobj >= 0
            obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
            lobj += obji * self.balance[i]

        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]

        loss = lbox + lobj + lcls
        loss_dict = dict(ss_box = lbox, ss_obj = lobj, ss_cls = lcls)
        return loss * bs, loss_dict

    def __call__(self, p, targets):
        if self.use_ota == False:
            loss, loss_dict = self.default_loss(p, targets)
        else:
            loss, loss_dict = self.ota_loss(p, targets)
        return loss, loss_dict
    
    def ota_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if targets.shape[1] > 6:
            reliable_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            bs, as_, gjs, gis, reliable_targets, anchors, tscores = self.assigner(p, reliable_targets, with_pseudo_scores=True)
            uc_bs, uc_as_, uc_gjs, uc_gis, uc_targets, uc_anchors, uc_tscores = self.assigner(p, uc_targets, with_pseudo_scores=True)
            pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    
            for i, pi in enumerate(p):
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
                tobj = torch.zeros_like(pi[..., 0], device=device)

                n = b.shape[0]
                if n:
                    ps = pi[b, a, gj, gi]

                    grid = torch.stack([gi, gj], dim=1)
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)
                    selected_tbox = reliable_targets[i][:, 2:6] * pre_gen_gains[i]
                    selected_tbox[:, :2] -= grid
                    iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                    lbox += (1.0 - iou).mean()

                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                    selected_tcls = reliable_targets[i][:, 1].long()
                    if self.nc > 1:
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)
                        t[range(n), selected_tcls] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)

                uc_b, uc_a, uc_gj, uc_gi = uc_bs[i], uc_as_[i], uc_gjs[i], uc_gis[i]
                n = uc_b.shape[0]
                if n:
                    if self.ignore_obj:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = -1
                    else:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = uc_tscores[i].type(tobj.dtype) 
                valid_mask = tobj >= 0
                obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
                lobj += obji * self.balance[i]
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        else:
            bs, as_, gjs, gis, targets, anchors = self.assigner(p, targets)
            pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    
            for i, pi in enumerate(p):
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
                tobj = torch.zeros_like(pi[..., 0], device=device)

                n = b.shape[0]
                if n:
                    ps = pi[b, a, gj, gi]

                    grid = torch.stack([gi, gj], dim=1)
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)
                    selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                    selected_tbox[:, :2] -= grid
                    iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                    lbox += (1.0 - iou).mean()

                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                    selected_tcls = targets[i][:, 1].long()
                    if self.nc > 1:
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)
                        t[range(n), selected_tcls] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)

                obji = self.BCEobj(pi[..., 4], tobj)
                lobj += obji * self.balance[i]
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]

        loss = lbox + lobj + lcls
        loss_dict = dict(ss_box = lbox, ss_obj = lobj, ss_cls = lcls)
        return loss * bs, loss_dict

