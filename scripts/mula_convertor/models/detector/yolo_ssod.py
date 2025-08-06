
import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.backbone.common import *
from models.head.yolov5_head import Detect
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
        self.loss_fn = self.cfg.Loss.type
        if self.loss_fn is not None:
            self.loss_fn = eval(self.loss_fn) if isinstance(self.loss_fn, str) else None

        self.model_type = 'yolov5'
        self.export = False

        self.det_8 = netD(cfg.Model.Neck.out_channels[0], cfg.Model.width_multiple)
        self.det_16 = netD(cfg.Model.Neck.out_channels[1], cfg.Model.width_multiple)
        self.det_32 = netD(cfg.Model.Neck.out_channels[2], cfg.Model.width_multiple)      

        self.check_head() 
        
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def check_head(self):
        m = self.head
        if isinstance(m, (Detect, RetinaDetect)):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.cfg.Model.ch, s, s))[0] ] )
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.initialize_biases()
        elif isinstance(m, (YoloV6Detect, YoloV8Detect)):
            m.inplace = self.inplace
            self.stride = torch.tensor(m.stride)
            m.initialize_biases()
            self.model_type = 'tal'
        elif isinstance(m, (YoloXDetect)):
            m.inplace = self.inplace
            self.stride = torch.tensor(m.stride)
            m.initialize_biases()
            self.model_type = 'yolox'

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)
    
    def forward_export(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)
        return out
    
    def _forward_once(self, x, profile=False, visualize=False):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)

        f8, f16, f32 = x
        feature = [f8, f16, f32]

        return out, feature

    def fuse(self):
        LOGGER.info('Fusing layers... ')
        for m in self.backbone.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        self.info()
        return self

    def autoshape(self):
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

class GradReverse(Function):
    def __init__(self):
        self.lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        result = x
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return (- grad_output)

def conv1x1(in_planes, out_planes, stride):
  "1x1 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class YoloX_netD_8(nn.Module):
    def __init__(self, ratio, context=False):
        super(YoloX_netD_8, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class YoloX_netD_16(nn.Module):
    def __init__(self, ratio, context=False):
        super(YoloX_netD_16, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class YoloX_netD_32(nn.Module):
    def __init__(self, ratio, context=False):
        super(YoloX_netD_32, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(256 * self.ratio), int(256* self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD(nn.Module):
    def __init__(self, channel, ratio, context=False):
        super(netD, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(channel * self.ratio), int(channel * self.ratio), stride=1)
        self.conv2 = conv1x1(int(channel * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_8(nn.Module):
    def __init__(self, channel, ratio, context=False):
        super(netD, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(channel * self.ratio), int(channel * self.ratio), stride=1)
        self.conv2 = conv1x1(int(channel * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_16(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_16, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(512 * self.ratio), int(512 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(512 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_32(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_32, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(1024 * self.ratio), int(1024* self.ratio), stride=1)
        self.conv2 = conv1x1(int(1024 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_res_8(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_res_8, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
class netD_res_16(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_res_16, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_res_32(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_res_32, self).__init__()
        self.ratio = ratio
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

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

