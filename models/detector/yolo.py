
import argparse
import sys
from copy import deepcopy
from pathlib import Path
from models.head.retina_head import RetinaDetect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.backbone.common import *
from models.backbone.experimental import *
from models.head.yolov5_head import Detect
from models.head.yolov7_head import IDetect
from models.head.yolov6_head import YoloV6Detect
from models.head.yolov8_head import YoloV8Detect
from models.head.yolox_head import YoloXDetect
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

from models.loss.loss import *
from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head

try:
    import thop
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml'):
        super().__init__()
        self.cfg = cfg
    
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.head = build_head(cfg)
        self.names = cfg.Dataset.names
        self.inplace = self.cfg.Model.inplace
        self.check_head()
     
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def check_head(self):
        m = self.head
        self.model_type = 'yolov5'
        if isinstance(m, (Detect, RetinaDetect, IDetect)):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.Tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.cfg.Model.ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.initialize_biases()

        elif isinstance(m, (YoloXDetect, YoloV6Detect, YoloV8Detect)):
            m.inplace = self.inplace
            self.stride = torch.Tensor(m.stride)
            m.initialize_biases()
            self.model_type = 'yolox'

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)
    
    def _forward_once(self, x, profile=False, visualize=False):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
      
        return x

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
                if isinstance(layer, QARepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
                if hasattr(layer, 'reparameterize'):
                    layer.reparameterize()
        for layer in self.neck.modules():
                if isinstance(layer, QARepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
                if hasattr(layer, 'reparameterize'):
                    layer.reparameterize()
        self.info()
        return self

    def autoshape(self):
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)
    print_args(FILE.stem, opt)
    set_logging()
    device = select_device(opt.device)

    model = Model(opt.cfg).to(device)
    model.train()

    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

