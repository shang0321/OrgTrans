import torch
import torch.nn as nn
from models.backbone.common import *
from utils.general import check_version

class RetinaDetect(nn.Module):
    stride = None
    def __init__(self, cfg):
        super(RetinaDetect, self).__init__()
        self.nc = cfg.Dataset.nc
        self.num_keypoints = cfg.Dataset.np
        self.cur_imgsize = [cfg.Dataset.img_size,cfg.Dataset.img_size]
        anchors = cfg.Model.anchors
        ch = []
        for out_c in cfg.Model.Neck.out_channels:
            ch.append(int(out_c * cfg.Model.width_multiple))
        self.no = self.nc + self.num_keypoints + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.stacked_convs = 4
        self.feat_channels = 256
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.feature = nn.ModuleList()
        for n in range(len(ch)):
            cls_convs = []
            reg_convs = []
            for i in range(self.stacked_convs):
                cls_convs += [Conv(256, 256, 3, 1, 1, 1, 'relu')]
                reg_convs += [Conv(256, 256, 3, 1, 1, 1, 'relu')]
            self.feature.append(Conv(256, 256, 3, 1, 1, 1, 'relu'))
            self.cls_convs.append(nn.Sequential(*cls_convs))
            self.reg_convs.append(nn.Sequential(*reg_convs))
        self.reg_m = nn.ModuleList(nn.Conv2d(x, 5 * self.na, 3, 1, 1) for x in ch)
        self.cls_m = nn.ModuleList(nn.Conv2d(x, 80 * self.na, 3, 1, 1) for x in ch)
        self.stride = cfg.Model.Head.strides
        self.export = False

    def initialize_biases(self, cf=None):
        for mi, s in zip(self.reg_m, self.stride):
            b = mi.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for mi, s in zip(self.cls_m, self.stride):
            b = mi.bias.view(self.na, -1)
            b.data[:, :] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def forward(self, x):
        z = []
        list_x = []
        for _ in x:
            list_x.append(_)
        x = list_x
        class_range = list(range(5 + self.nc))
        for i in range(self.nl):
            feat = self.feature[i](x[i])
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            reg = self.reg_m[i](reg_feat)
            cls = self.cls_m[i](cls_feat)
            reg_list = torch.split(reg, 5, dim=1)
            cls_list = torch.split(cls, 80, dim=1)
            x[i] = torch.cat((reg_list[0], cls_list[0], reg_list[1], cls_list[1], reg_list[2], cls_list[2]), 1)
            bs, _, ny, nx = x[i].shape
            if hasattr(self, 'export') and self.export:
                x[i] = x[i].view(bs, self.na, self.no, -1).permute(0, 1, 3, 2)
                z.append(x[i])
                continue

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid_old(nx, ny, i)

                y = torch.full_like(x[i], 0)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]

                z.append(y.view(bs, -1, self.no))

        if hasattr(self, 'export') and self.export:
            return tuple(z)

        return x if self.training  else (torch.cat(z, 1), x)

    def post_process(self, x):
        z = []
        for i in range(self.nl):
            ny,nx = int(self.cur_imgsize[0]/self.stride[i]),int(self.cur_imgsize[1]/self.stride[i])
            bs, _, nyxnx, _ = x[i].shape
            x[i] = x[i].view(bs, self.na, ny, nx, self.no).contiguous()

            if not self.training:
                self.grid[i], self.anchor_grid[i] = self._make_grid_old(nx, ny, i)

                y = torch.full_like(x[i], 0)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                class_range = list(range(5+self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else [torch.cat(z, 1), x]

    def _make_grid_old(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid