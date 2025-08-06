
import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.head.yolov5_head import Detect
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info
   
from models.loss.loss import ComputeLoss
from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head
import logging
import torch.nn as nn
import torch
import math

try:
    import thop
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

def check_anchor_order(m):
    a = m.anchors.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml'):
        super().__init__()
        self.cfg = cfg
    
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.head = build_head(cfg)
        self.names = cfg.Dataset.names
        self.inplace = self.cfg.Model.inplace
        self.loss_fn = self.cfg.Loss.type
        if self.loss_fn is not None:
            self.loss_fn = eval(self.loss_fn) if isinstance(self.loss_fn, str) else None

        m = self.head
        self.model_type = 'yolov5'
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.Tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, cfg.Model.ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)
    
    def _forward_once(self, x, profile=False, visualize=False):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
      
        return x

    def _initialize_biases(self, cf=None):
        m = self.head
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        LOGGER.info('Fusing layers... ')
        for m in self.backbone.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse

        for m in self.neck.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        
        for layer in self.backbone.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
        for layer in self.neck.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
        self.info()
        return self

    def autoshape(self):
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

