
import logging
import torch
import numpy as np

def center_to_corner(cxcy):
    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)

def corner_to_center(xy):
    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=-1)

def find_jaccard_overlap(set_1, set_2, eps=1e-5):

    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps

    return intersection / union

def find_intersection(set_1, set_2):

    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

def make_center_anchors(anchors_wh, grid_size=80, device='cpu'):

    grid_arange = torch.arange(grid_size)
    xx, yy = torch.meshgrid(grid_arange, grid_arange)
    xy = torch.cat((torch.unsqueeze(xx, -1), torch.unsqueeze(yy, -1)), -1) + 0.5

    wh = torch.tensor(anchors_wh)

    xy = xy.view(grid_size, grid_size, 1, 2).expand(grid_size, grid_size, 3, 2).type(torch.float32)
    wh = wh.view(1, 1, 3, 2).expand(grid_size, grid_size, 3, 2).type(torch.float32)
    center_anchors = torch.cat([xy, wh], dim=3).to(device)

    return center_anchors

def get_imitation_mask(x, targets, det_anchors, stride, iou_factor=0.5):
        det_anchors = np.array(det_anchors).reshape(-1, 2)/stride
        num_anchors = len(det_anchors)
        out_size = x.size(2)
        batch_size = x.size(0)
        device = targets.device

        mask_batch = torch.zeros([batch_size, out_size, out_size])
        
        if not len(targets):
            return mask_batch
        
        gt_boxes = [[] for i in range(batch_size)]
        for i in range(len(targets)):
            gt_boxes[int(targets[i, 0].data)] += [targets[i, 2:].clone().detach().unsqueeze(0)]
        
        max_num = 0
        for i in range(batch_size):
            max_num = max(max_num, len(gt_boxes[i]))
            if len(gt_boxes[i]) == 0:
                gt_boxes[i] = torch.zeros((1, 4), device=device)
            else:
                gt_boxes[i] = torch.cat(gt_boxes[i], 0)
        
        for i in range(batch_size):
            if max_num - gt_boxes[i].size(0):
                gt_boxes[i] = torch.cat((gt_boxes[i], torch.zeros((max_num - gt_boxes[i].size(0), 4), device=device)), 0)
            gt_boxes[i] = gt_boxes[i].unsqueeze(0)

        gt_boxes = torch.cat(gt_boxes, 0)
        gt_boxes *= out_size
        
        center_anchors = make_center_anchors(anchors_wh=det_anchors, grid_size=out_size, device=device)
        anchors = center_to_corner(center_anchors).view(-1, 4)
        
        gt_boxes = center_to_corner(gt_boxes)

        mask_batch = torch.zeros([batch_size, out_size, out_size], device=device)

        for i in range(batch_size):
            num_obj = gt_boxes[i].size(0)
            if not num_obj:
                continue
             
            IOU_map = find_jaccard_overlap(anchors, gt_boxes[i], 0).view(out_size, out_size, num_anchors, num_obj)
            max_iou, _ = IOU_map.view(-1, num_obj).max(dim=0)
            mask_img = torch.zeros([out_size, out_size], dtype=torch.int64, requires_grad=False).type_as(x)
            threshold = max_iou * iou_factor

            for k in range(num_obj):

                mask_per_gt = torch.sum(IOU_map[:, :, :, k] > threshold[k], dim=2)

                mask_img += mask_per_gt

                mask_img += mask_img
            mask_batch[i] = mask_img

        mask_batch = mask_batch.clamp(0, 1)
        return mask_batch