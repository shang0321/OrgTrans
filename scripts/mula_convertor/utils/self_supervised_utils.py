
import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import random
from utils.general import xyxy2xywh, xywh2xyxy, xywhn2xyxy, non_max_suppression, box_iou
from utils.plots import plot_images, plot_labels, plot_results,  plot_images_debug, output_to_target
from utils.torch_utils import time_sync
from models.loss.iou_loss import bbox_overlaps
import copy

try:
    import thop
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=dim)
    return output
class LabelMatch:
    def __init__(self, num_classes):
        self.nc = num_classes
        self.cls_thr = [0.1] * self.nc
        self.percent = 0.2
        self.max_cls_per_img = 20
        self.queue_num = 128 * 100
        self.queue_len = self.queue_num * self.max_cls_per_img
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.boxes_per_image_gt = 7
        self.cls_ratio_gt = np.array([1.0/self.nc] * self.nc)
        self.score_list = [[0]] * self.nc

    def update_cls_thr(self):
        for cls in range(self.nc):
            s = self.score_list[cls].sort()
            self.cls_thr[cls] = max(0.1, s[0])
        info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(self.cls_thr)])
        LOGGER.info(f'update score thr (positive): {info}')
    
    def clean_score_list(self):
        self.score_list = [[0]] * self.nc
    
    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        n_img, _, height, width = target_imgs.shape
        lb = []
      
        pseudo_out = non_max_suppression(out, conf_thres=0.1, iou_thres=0.65, labels=lb, multi_label=True)

        refine_out = []
        print('score_list:', self.score_list)
        for i, o in enumerate(pseudo_out):
            for *box, conf, cls in o.cpu().numpy():
                self.score_list[int(cls)].append(float(conf))
                if conf > self.cls_thr[int(cls)]:
                    refine_out.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
        refine_out = np.array(refine_out)
        target_out_targets = torch.tensor(refine_out)
        target_shape = target_out_targets.shape
        target_out_targets_perspective = []
        invalid_target_shape = True
        if target_shape[0] == 0:
            return target_out_targets_perspective, target_imgs, invalid_target_shape 

        if(target_shape[0] > 0 and target_shape[1] > 6):
            for i, img in enumerate(target_imgs_ori):
                image_targets = refine_out[refine_out[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                M_select = M_s[M_s[:, 0] == i, :]
                M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
               
                image_targets = np.array(image_targets_random)
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])
                    image_targets[:, [3, 5]] /= height
                    image_targets[:, [2, 4]] /= width
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
            invalid_target_shape = False
        return target_out_targets_perspective, target_imgs, invalid_target_shape

class FairPseudoLabel:
    def __init__(self, conf_thresh, valid_thresh, nms_conf_thresh, nms_iou_thresh, debug=False):
        self.conf_thresh = conf_thresh
        self.valid_thresh = valid_thresh
        self.nms_conf_thresh = nms_conf_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.debug = debug

    def online_label_transform_with_image(self, img, targets, M, s, ud, lr, segments=(), border=(0, 0), perspective=0.0):
        if isinstance(img, torch.Tensor):
                img = img.cpu().float().numpy()
                img = img.transpose(1, 2, 0) * 255.0
                img = img.astype(np.uint8)
        height = img.shape[0] + border[0] * 2
        width = img.shape[1] + border[1] * 2
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        if ud == 1:
            img = np.flipud(img)
        if lr == 1:
            img = np.fliplr(img)
        img = torch.from_numpy(img.transpose(2, 0, 1)/255.0)

        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))
            if use_segments:
                segments = resample_segments(segments)
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

                    new[i] = segment2box(xy, width, height)

            else:
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
                xy = xy @ M.T
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets

    def create_pseudo_label_online(self, out, target_imgs, M_s, target_imgs_ori, gt=None):
        n_img, _, height, width = target_imgs.shape
        lb = []
        target_out_targets_perspective = []
        invalid_target_shape = True

        pseudo_out = non_max_suppression(out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb, multi_label=True)
        pseudo_target = output_to_target(pseudo_out)
        pseudo_out_list = []
        for out_tensor in pseudo_out:
            out_array = out_tensor.detach().cpu().numpy()
            if out_array.shape[0] == 0:
                continue
            else:
                for out_instance in out_array:
                    pseudo_out_list.append(out_instance[4])

        if pseudo_out_list == []:
            return target_out_targets_perspective, target_imgs, invalid_target_shape 
        else:
            pseudo_out = np.array(pseudo_out_list)

        out = non_max_suppression(out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb)
        out = [out_tensor.detach() for out_tensor in out]
        target_out_np = output_to_target(out)
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape
        total_t1 = time_sync()
        if(target_shape[0] > 0 and target_shape[1] > 6):
            for i, img in enumerate(target_imgs):
             
                image_targets = target_out_targets[target_out_targets[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                M_select = M_s[M_s[:, 0] == i, :]
                M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
               
                if 1:
                    image_targets = np.array(image_targets_random)
                else:
                    image_targets = np.array(image_targets[:, 1:])
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])
                    image_targets[:, [3, 5]] /= height
                    image_targets[:, [2, 4]] /= width
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
            invalid_target_shape = False
        return target_out_targets_perspective, target_imgs, invalid_target_shape

    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        n_img, _, height, width = target_imgs.shape
        lb = []
      
        target_out_targets_perspective = []
        invalid_target_shape = True

        pseudo_out = non_max_suppression(out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb, multi_label=True)
        pseudo_target = output_to_target(pseudo_out)
        pseudo_out_list = []
        for out_tensor in pseudo_out:
            out_array = out_tensor.detach().cpu().numpy()
            if out_array.shape[0] == 0:
                continue
            else:
                for out_instance in out_array:
                    pseudo_out_list.append(out_instance[4])

        if pseudo_out_list == []:
            return target_out_targets_perspective, target_imgs, invalid_target_shape 
        else:
            pseudo_out = np.array(pseudo_out_list)
            mean_conf = float(np.mean(pseudo_out))
            valid_index = pseudo_out > mean_conf
            valid_num = np.sum(valid_index)
            total_num = pseudo_out.shape[0]
            valid_ratio = valid_num * 1.0 / total_num
        
            for i, valid in enumerate(valid_index):
                if not valid:
                    pseudo_target[i][1] = 3
            if(mean_conf < self.conf_thresh or valid_ratio < self.valid_thresh):
                return target_out_targets_perspective, target_imgs, invalid_target_shape 

        out = non_max_suppression(out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb)
        out = [out_tensor.detach() for out_tensor in out]
        target_out_np = output_to_target(out)
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape
        total_t1 = time_sync()
        if(target_shape[0] > 0 and target_shape[1] > 6):
            for i, img in enumerate(target_imgs_ori):
                image_targets = target_out_targets[target_out_targets[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                M_select = M_s[M_s[:, 0] == i, :]
                M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
               
                image_targets = np.array(image_targets_random)
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])
                    image_targets[:, [3, 5]] /= height
                    image_targets[:, [2, 4]] /= width
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
            invalid_target_shape = False
            if self.debug:
                if RANK in [-1 ,0]:
                    draw_image = plot_images(copy.deepcopy(target_imgs), target_out_targets_perspective, None, '/mnt/bowen/EfficientTeacher/unbias_teacher_pseudo_label.jpg')            

        return target_out_targets_perspective, target_imgs, invalid_target_shape

    def create_pseudo_label_online_with_extra_teachers(self, out, extra_teacher_outs, target_imgs, M, \
     extra_teacher_class_idxs):
        n_img, _, height, width = target_imgs.shape
        lb = []

        target_out_targets_perspective = []
        invalid_target_shape = True
        pseudo_out = non_max_suppression(out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb, multi_label=True)

        for teacher_idx, teacher_out in enumerate(extra_teacher_outs):
            teacher_pseudo_out = non_max_suppression(teacher_out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb, multi_label=True)
            for i, o in enumerate(pseudo_out):
                pseudo_out_one_img = teacher_pseudo_out[i]
                if pseudo_out_one_img.shape[0] > 0 :
                    for each in pseudo_out_one_img:
                        origin_class_idx = int(each[5].cpu().item() )
                        if origin_class_idx in extra_teacher_class_idxs[teacher_idx]:
                            each[5] = float(extra_teacher_class_idxs[teacher_idx][origin_class_idx])

                x = torch.cat([o, pseudo_out_one_img])
                c = x[:, 5:6] * 0
                boxes, scores = x[:, :4] + c, x[:, 4] 
                index = torchvision.ops.nms(boxes, scores, self.nms_iou_thresh)
                pseudo_out[i] = x[index]

        pseudo_target = output_to_target(pseudo_out)
        pseudo_out_list = []
        for out_tensor in pseudo_out:
            out_array = out_tensor.detach().cpu().numpy()
            if out_array.shape[0] == 0:
                continue
            else:
                for out_instance in out_array:
                    pseudo_out_list.append(out_instance[4])
        if pseudo_out_list == []:
            return target_out_targets_perspective, target_imgs, invalid_target_shape 
        else:
            pseudo_out = np.array(pseudo_out_list)
            mean_conf = float(np.mean(pseudo_out))
            valid_index = pseudo_out > mean_conf
            valid_num = np.sum(valid_index)
            total_num = pseudo_out.shape[0]
            valid_ratio = valid_num * 1.0 / total_num
        
            for i, valid in enumerate(valid_index):
                if not valid:
                    pseudo_target[i][1] = 14
            if(mean_conf < self.conf_thresh or valid_ratio < self.valid_thresh):
                return target_out_targets_perspective, target_imgs, invalid_target_shape 

        out = non_max_suppression(out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb, multi_label=True)
        out = [out_tensor.detach() for out_tensor in out]
        for teacher_idx, teacher_out in enumerate(extra_teacher_outs):
            teacher_pseudo_out = non_max_suppression(teacher_out, conf_thres=self.nms_conf_thresh, iou_thres=self.nms_iou_thresh, labels=lb, multi_label=True)
            for i, o in enumerate(out):
                pseudo_out_one_img = teacher_pseudo_out[i]
                if pseudo_out_one_img.shape[0] > 0 :
                    for each in pseudo_out_one_img:
                        origin_class_idx = int(each[5].cpu().item() )
                        if origin_class_idx in extra_teacher_class_idxs[teacher_idx]:
                            each[5] = float(extra_teacher_class_idxs[teacher_idx][origin_class_idx])

                x = torch.cat([o, pseudo_out_one_img])
                c = x[:, 5:6] * 0
                boxes, scores = x[:, :4] + c, x[:, 4] 
                index = torchvision.ops.nms(boxes, scores, self.nms_iou_thresh)
                out[i] = x[index]

        target_out_np = output_to_target(out)
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape
        if(target_shape[0] > 0 and target_shape[1] > 6):
            for i, img in enumerate(target_imgs):
                image_targets = target_out_targets[target_out_targets[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M[i][0], M[i][1])
                image_targets = np.array(image_targets_random)
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])
                    image_targets[:, [3, 5]] /= height
                    image_targets[:, [2, 4]] /= width
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        if target_shape[0] > 0:
            invalid_target_shape = False
        return target_out_targets_perspective, target_imgs, invalid_target_shape

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)

def online_random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):

    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

                new[i] = segment2box(xy, width, height)

        else:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def online_label_transform(img, targets, M, s, segments=(), border=(0, 0), perspective=0.0):
        height = img.shape[1] + border[0] * 2
        width = img.shape[2] + border[1] * 2

        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))
            if use_segments:
                segments = resample_segments(segments)
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

                    new[i] = segment2box(xy, width, height)

            else:
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
                xy = xy @ M.T
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets

def check_pseudo_label_with_gt(detections, labels, iouv=torch.tensor([0.5])):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    gt = labels[:, 2:6]
    pseudo = detections[:, 2:6]
    gt *= torch.tensor([640, 640] * 2)
    pseudo *= torch.tensor([640, 640] * 2)
    gt = xywh2xyxy(gt)
    pseudo = xywh2xyxy(pseudo)
    iou = box_iou(gt, pseudo)
    correct_class = labels[:, 1:2] == detections[:, 1]
    correct_image = labels[:, 0:1] == detections[:, 0]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class & correct_image)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    
    hit_rate = np.sum(correct, 0) * 1.0/ labels.shape[0]
    return hit_rate

def check_pseudo_label(detections, labels, iouv=torch.tensor([0.5])):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    gt = labels[:, 2:6]
    pseudo = detections[:, 2:6]
    gt *= torch.tensor([640, 640] * 2)
    pseudo *= torch.tensor([640, 640] * 2)
    gt = xywh2xyxy(gt)
    pseudo = xywh2xyxy(pseudo)
    iou = box_iou(gt, pseudo)
    correct_class = labels[:, 1:2] == detections[:, 1]
    correct_image = labels[:, 0:1] == detections[:, 0]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class & correct_image)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    
    hit_rate = np.sum(correct, 0) * 1.0/ labels.shape[0]
    return hit_rate
   
if __name__ == '__main__':
    label_match = LabelMatch(0.1, 80)