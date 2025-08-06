import copy

from .yolov5_head import Detect

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.Model.Head.name
    if name == "YoloV5":
        return Detect(head_cfg)
    else:
        raise NotImplementedError
