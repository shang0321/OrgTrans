class ComputeStudentMatchLoss:
    def __init__(self, model, hyp, ssod_hyp):
        self.model = model
        self.hyp = hyp
        self.ssod_hyp = ssod_hyp
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=model.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=model.device))
        self.box_loss = BoxLoss(model)
        self.uncertainty_threshold = 0.5
        self.ema_uncertainty = 0.0
        self.ema_alpha = 0.99
        self.sup_weight = 1.0
        self.unsup_weight = 0.5
        self.uncertainty_weight = 0.1

    def __call__(self, preds, targets, imgs, pseudo_targets=None):
        device = preds[0].device
        loss = torch.zeros(3, device=device)

        tcls, tbox, indices, anchors = self.build_targets(preds, targets)
        sup_loss = torch.zeros(3, device=device)
        for i, pred in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pred[..., 0], device=device)
            n = b.shape[0]
            if n:
                ps = pred[b, a, gj, gi]
                if self.ssod_hyp.get('pseudo_label_with_cls', False):
                    t = torch.full_like(ps[:, 5:], 0.0, device=device)
                    t[range(n), tcls[i]] = 1.0
                    sup_loss[2] += self.BCEcls(ps[:, 5:], t)
                if self.ssod_hyp.get('pseudo_label_with_bbox', False):
                    sup_loss[0] += self.box_loss(ps[:, :4], tbox[i])
                if self.ssod_hyp.get('pseudo_label_with_obj', False):
                    tobj[b, a, gj, gi] = 1.0
                    sup_loss[1] += self.BCEobj(pred[..., 4], tobj)

        unsup_loss = torch.zeros(3, device=device)
        if pseudo_targets is not None:
            ptcls, ptbox, pindices, panchors = self.build_targets(preds, pseudo_targets)
            for i, pred in enumerate(preds):
                b, a, gj, gi = pindices[i]
                tobj = torch.zeros_like(pred[..., 0], device=device)
                n = b.shape[0]
                if n:
                    ps = pred[b, a, gj, gi]
                    uncertainty = self.compute_uncertainty(ps)
                    weight = torch.exp(-self.uncertainty_weight * uncertainty)
                    
                    if self.ssod_hyp.get('pseudo_label_with_cls', False):
                        t = torch.full_like(ps[:, 5:], 0.0, device=device)
                        t[range(n), ptcls[i]] = 1.0
                        unsup_loss[2] += weight * self.BCEcls(ps[:, 5:], t)
                    if self.ssod_hyp.get('pseudo_label_with_bbox', False):
                        unsup_loss[0] += weight * self.box_loss(ps[:, :4], ptbox[i])
                    if self.ssod_hyp.get('pseudo_label_with_obj', False):
                        tobj[b, a, gj, gi] = 1.0
                        unsup_loss[1] += weight * self.BCEobj(pred[..., 4], tobj)

        total_loss = self.sup_weight * sup_loss.sum() + self.unsup_weight * unsup_loss.sum()
        return total_loss

    def compute_uncertainty(self, preds):
        cls_probs = torch.sigmoid(preds[..., 5:])
        entropy = -torch.sum(cls_probs * torch.log(cls_probs + 1e-10), dim=-1)
        
        box_var = torch.var(preds[..., :4], dim=-1)
        
        obj_uncertainty = torch.abs(torch.sigmoid(preds[..., 4]) - 0.5) * 2
        
        uncertainty = (entropy + box_var + obj_uncertainty) / 3
        
        self.ema_uncertainty = self.ema_alpha * self.ema_uncertainty + (1 - self.ema_alpha) * uncertainty.mean().item()
        
        return uncertainty 