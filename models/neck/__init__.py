
import copy

from .yolov5_neck import YoloV5Neck
from .yolov6_neck import YoloV6Neck
from .yolov7_neck import YoloV7Neck
from .yolov8_neck import YoloV8Neck

def build_neck(cfg,nas_arch=None,in_channels=None):
    fpn_cfg = copy.deepcopy(cfg)
    name = cfg.Model.Neck.name
    if name == "YoloV5":
        return YoloV5Neck(fpn_cfg)
    elif name == "YoloV6":
        return YoloV6Neck(fpn_cfg)
    elif name == "YoloV7":
        return YoloV7Neck(fpn_cfg)
    elif name == "YoloV8":
        return YoloV8Neck(fpn_cfg)
    else:
        raise NotImplementedError
