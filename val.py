
import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.datasets import create_dataloader
from utils.metrics import ap_per_class, ConfusionMatrix, oks_iou
from utils.general import coco80_to_coco91_class, check_dataset, check_img_size, check_requirements, \
    check_suffix, check_yaml, box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, \
    increment_path, colorstr, print_args, non_max_suppression_lmk_and_bbox, scale_coords_landmarks
from utils.plots import plot_images
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.profile import profile
from utils.torch_utils import is_parallel
import copy
from configs.defaults import get_cfg

from utils.metrics import NMEMeter
from utils.detect_multi_backend import DetectMultiBackend

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

def process_batch_oks(detections, labels, iouv, num_points):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    correct_class = labels[:, 0:1] == detections[:, 5]
    ious = oks_iou(labels, detections, num_points)
    ious = torch.from_numpy(ious).to(iouv.device)

    for i in range(len(iouv)):
        x = torch.where((ious >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct

def process_batch_old(detections, labels, iouv):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        task='val',
        device='',
        single_cls=False,
        augment=False,
        verbose=False,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        save_json=False,
        project=ROOT / 'runs/val',
        name='exp',
        exist_ok=False,
        half=True,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        model_post=None,
        eval_num = -1,
        cfg = None,
        val_ssod = False,
        num_points = 0,
        val_kp = False,
        val_dp1000 = False,
        dnn=False,
        names = {}
        ):
    training = model is not None
    if training:
        device = next(model.parameters()).device
        pt, jit, engine = True, False, False
        half &= device.type != 'cpu'
        model.half() if half else model.float()
    else:
        device = select_device(device, batch_size=batch_size)

        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine, is_magicmind = model.stride, model.pt, model.jit, model.engine, model.magicmind
        gs = max(int(model.stride), 32)
        imgsz = check_img_size(imgsz, s=gs)

        half = model.fp16
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1

        if cfg != '':
            val_cfg = get_cfg()
            val_cfg.merge_from_file(cfg)
            data = {}
            data['val'] = val_cfg.Dataset.val
            data['nc'] = val_cfg.Dataset.nc
            data['names'] = val_cfg.Dataset.names
            val_kp = val_cfg.Dataset.val_kp
        else:
            data = check_dataset(data)
            val_cfg = None

        if val_ssod:
            pass
        elif pt:
            model_profile = copy.deepcopy(model)
            flops, params =  profile(model_profile.module if is_parallel(model_profile) else model_profile, (torch.ones((1, 3, imgsz, imgsz)).to(device),1), clever=True)
            print("Flops {} Params {}".format(flops, params))
        else:
            pass

    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    if not training:
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True, cfg=val_cfg,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    try:
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    except:
        names = names
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        if batch_i==eval_num:
            break
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        if pt:
            img /= 255.0
        elif is_magicmind:
            pass
        else:
            img /= 255.0   
        targets = targets.to(device)
        nb, _, height, width = img.shape
        t2 = time_sync()
        dt[0] += t2 - t1

        if val_ssod:
            outputs, sup_feats = model(img, augment=augment) 
        else:
            outputs = model(img, augment=augment)
        if model_post is not None:
            model_post.cur_imgsize = img.shape[2:]
            if is_magicmind:
                outputs = model_post.post_process_v2(outputs)
            else:
                outputs = model_post.post_process(outputs)
        dt[1] += time_sync() - t2

        if not isinstance(outputs, list):
            if isinstance(outputs, tuple):
                out = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[0][0]
            else:
                out = outputs
        else:
            if len(outputs) >= 2:
                out = outputs[0][0] if isinstance(outputs[0], (tuple, list)) else outputs[0]
                train_out = outputs[1]
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1]
            else:
                out = outputs[0][0] if isinstance(outputs[0], (tuple, list)) else outputs[0]
        
        if not isinstance(out, torch.Tensor):
            raise ValueError(f"Expected tensor output but got {type(out)}. Model output structure: {type(outputs)}")
            
        if val_kp:
            targets[:, 2:2+2*(2 + num_points)] *= torch.Tensor([width, height] * (2 + num_points)).to(device)
        else:
            targets[:, 2:6] *= torch.Tensor([width, height] * 2).to(device)
        lb = [targets[targets[:, 0] == i, 1:6] for i in range(nb)] if save_hybrid else []
        t3 = time_sync()
        if num_points > 0:
            out = non_max_suppression_lmk_and_bbox(out, conf_thres, iou_thres, labels=lb, agnostic=single_cls, num_points=num_points)
        else:
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            if num_points > 0:
                scale_coords_landmarks(img[si].shape[1:], predn[:, -1 - num_points * 2:-1], shape, num_points, shapes[si][1])

            if nl:
                if num_points > 0 and val_kp:
                    scale_coords_landmarks(img[si].shape[1:], labels[:, 5:5+num_points * 2], shape, num_points, shapes[si][1])
                    correct = process_batch_oks(predn, labels, iouv, num_points)
                else:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = process_batch(predn, labelsn, iouv)

                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'
            if val_kp:
                Thread(target=plot_images, args=(img, targets, paths, f, num_points, names), daemon=True).start()
            else:
                Thread(target=plot_images, args=(img, targets, paths, f, 0, names), daemon=True).start()

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, cls_thr = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        cls_thr = []
        nt = torch.zeros(1)
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x / seen * 1E3 for x in dt)
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = str('/mnt/bowen/exp/instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.json")
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    model.float()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    if val_ssod:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, cls_thr
    else:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '/home/shangzhipeng/data_YOLO/C_yolo/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '/home/shangzhipeng/project/efficientteacher/ssod_runs/L-c/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--val-ssod',  action='store_true', help='trigger when val semi-supervised') 
    parser.add_argument('--num-points',  type=int, default=0, help='num of keypoints') 
    parser.add_argument('--cfg', type=str, default='', help='The config file used for validation') 
    parser.add_argument('--val-dp1000',  action='store_true', help='trigger when val dp1000 model') 
    opt = parser.parse_args()
    if opt.cfg == '':
        opt.data = check_yaml(opt.data)
        opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    set_logging()
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
