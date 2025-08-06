
import copy

from .yolov5_neck import YoloV5Neck
from .yolov7_neck import YoloV7Neck

def build_neck(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = cfg.Model.Neck.name
    if name == "YoloV5":
        return YoloV5Neck(fpn_cfg)
    elif name == "YoloV7":
        return YoloV7Neck(fpn_cfg)
    else:
        raise NotImplementedError
