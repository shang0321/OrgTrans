
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
from utils.general import clip_coords, xyxy2xywh, xywh2xyxy, xywhn2xyxy, non_max_suppression, box_iou
from utils.general import non_max_suppression_ssod
from utils.plots import plot_images_ssod, plot_images, plot_labels,  output_to_target_ssod
from utils.torch_utils import time_sync
from utils.self_supervised_utils import online_label_transform
import copy
import matplotlib.pyplot as plt
import sklearn.mixture as skm

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

class LabelMatch(nn.Module):
    def __init__(self, cfg, target_data_len, label_num_per_img, cls_ratio_gt):
        super().__init__()
        self.nc = cls_ratio_gt.shape[0]
        self.multi_label = cfg.SSOD.multi_label
        self.nms_conf_thres = cfg.SSOD.nms_conf_thres
        self.nms_iou_thres = cfg.SSOD.nms_iou_thres
        self.cls_thr_high = [cfg.SSOD.ignore_thres_high] * self.nc
        self.cls_thr_low = [cfg.SSOD.ignore_thres_low] * self.nc
        self.max_cls_per_img = 10
        self.boxes_per_image_gt = 7
        self.queue_num = 128
        self.queue_len = self.queue_num * self.max_cls_per_img
        self.cls_ratio_gt = cls_ratio_gt 
        self.register_buffer("score_queue" , torch.zeros(self.nc, self.queue_len))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.score_list = [[] for _ in range(self.nc)]
        self.cls_list = [range(self.nc)]
        self.start_update = False
        self.ignore_thres_low = cfg.SSOD.ignore_thres_low
        self.ignore_thres_high = cfg.SSOD.ignore_thres_high
        self.score_list_epoch = [[] for _ in range(self.nc)]
        self.debug = cfg.SSOD.debug
        self.resample_high_percent = cfg.SSOD.resample_high_percent
        self.resample_low_percent = cfg.SSOD.resample_low_percent
        self.names = cfg.Dataset.names
        self.num_points = cfg.Dataset.np
        self.target_data_len = target_data_len
        self.anno_num_per_img = label_num_per_img * 3
        self.cls_num_list = [0 for _ in range(self.nc)]
        self.count = 0
        self.pse_count = 0

        self.cls_tmp = np.zeros(self.nc)
        self.cls_num_total = np.zeros(self.nc)
    
    def _dequeue_and_enqueue(self, score):
        print('check device:', score[0, :])
        if dist.is_initialized():
            score = concat_all_gather(score, 1) 
        print('result:', score[0, :])
        batch_size = score.shape[1]
        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0
        if ptr + batch_size > self.queue_len:
            lens = self.queue_len - ptr
            self.score_queue[:, ptr:ptr + lens] = score[:, :lens]
        else:
            self.score_queue[:, ptr:ptr + batch_size] = score
        if (not self.start_update) and ptr + batch_size >= self.queue_len:
            self.start_update = True
        ptr = (ptr + batch_size) % self.queue_len
        self.queue_ptr[0] = ptr
    
    def update(self, labels, n=1, pse_n=1):
        self.count += n
        self.pse_count += pse_n
        for l in labels:
            self.cls_tmp[int(l[1:2])] += 1
    
    def gmm_policy(self, scores, given_gt_thr=0.5, policy='high'):
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                        scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr
    
    def update_epoch_cls_thr(self, epoch):
        print('cls_tmp:', self.cls_ratio_gt)
        print('count:', self.count)
        print('target_data_len:', self.pse_count)

        for cls in range(self.nc):
            single_cls_score = self.score_list_epoch[cls]
            single_cls_score.sort(reverse=True)
            self.cls_num_total[cls] += len(single_cls_score)
            max_pseudo_label_num = int(self.cls_num_total[cls] / (epoch + 1))
            if len(single_cls_score) == 0:
                self.cls_thr_high[cls] = self.ignore_thres_high
                self.cls_thr_low[cls] = self.ignore_thres_low
                pos_loc_high = -1
                pos_loc_low = -1
            else:
                pos_loc_high = int(len(single_cls_score) * self.resample_high_percent)
                pos_loc_low = min(max_pseudo_label_num, int(len(single_cls_score) * self.resample_low_percent))
                gmm_thr = self.gmm_policy(np.array(single_cls_score), given_gt_thr=0.0, policy='high')
                self.cls_thr_high[cls] = gmm_thr
                self.cls_thr_low[cls] = max(self.ignore_thres_low, single_cls_score[pos_loc_low])
            MIN_LOW = 0.1
            MAX_HIGH = 0.9
            self.cls_thr_low[cls] = max(MIN_LOW, min(self.cls_thr_low[cls], MAX_HIGH))
            self.cls_thr_high[cls] = min(MAX_HIGH, max(self.cls_thr_high[cls], MIN_LOW))
            LOGGER.info(f'{len(single_cls_score)} {max_pseudo_label_num} {pos_loc_high}:{self.cls_thr_high[cls]} {pos_loc_low}:{self.cls_thr_low[cls]}')
        self.score_list_epoch = [[] for _ in range(self.nc)]
        self.cls_tmp = np.zeros(self.nc)
        self.count = 0
        self.pse_count = 0

    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        n_img, _, height, width = target_imgs.shape
        lb = []
      
        pseudo_out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres,\
             labels=lb, num_points=self.num_points, multi_label=self.multi_label)

        refine_out = []
        score_list_batch = [[] for _ in range(self.nc)]
        for i, o in enumerate(pseudo_out):
            for *box, conf, cls, obj_conf, cls_conf in o.cpu().numpy():
                score_list_batch[int(cls)].append(float(conf))
                refine_out.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, obj_conf, cls_conf])
                self.score_list_epoch[int(cls)].append(float(conf))

        score_list = [[] for _ in range(self.nc)]
        for c in range(len(score_list_batch)):
                if len(score_list_batch[c]) < self.max_cls_per_img:
                    score_list_batch[c].extend([0.0] * (self.max_cls_per_img - len(score_list_batch[c])))
                    score_list_batch[c].sort(reverse=True)
                else:
                    score_list_batch[c].sort(reverse=True)
                    score_list_batch[c] = score_list_batch[c][:self.max_cls_per_img]
        try:
            score = np.array(score_list_batch).astype(np.float32)
            score = torch.from_numpy(score).contiguous()
        except:
            for fp in score_list:
                print(len(fp))
                if len(fp) != 160:
                    print(fp)

        refine_out = np.array(refine_out)
        target_out_targets = torch.tensor(refine_out)
        target_shape = target_out_targets.shape
        target_out_targets_perspective = []
        invalid_target_shape = True
        if target_shape[0] == 0:
            return target_out_targets_perspective, invalid_target_shape

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
                    image_targets[:, 2:6] = image_targets[:, 2:6].clip(0, 1)
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
            invalid_target_shape = False
            if self.debug and RANK in [-1 ,0]:
                draw_image = plot_images_ssod(copy.deepcopy(target_imgs), target_out_targets_perspective, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label_ea.jpg', names=self.names)            
                draw_image = plot_images(copy.deepcopy(target_imgs), gt, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_gt_ea.jpg', names=self.names)            
        return target_out_targets_perspective, invalid_target_shape
        