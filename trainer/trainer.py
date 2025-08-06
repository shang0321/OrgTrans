
import logging
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD, lr_scheduler
from tqdm import tqdm

from models.backbone.experimental import attempt_load
from models.detector.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, init_seeds, \
    strip_optimizer, check_img_size, check_suffix, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from models.loss.loss import ComputeLoss, ComputeNanoLoss
from models.loss.yolox_loss import ComputeFastXLoss
from utils.plots import plot_labels
from utils.torch_utils import ModelEMA, de_parallel, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.metrics import MetricMeter, fitness
from utils.loggers import Loggers
from contextlib import redirect_stdout
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE):
        self.cfg = cfg
        self.set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)
        self.opt_scales = None
        ckpt = self.build_model(cfg, device)
        self.build_optimizer(cfg,ckpt=ckpt)

        self.build_dataloader(cfg, callbacks)
       
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                f'Using {self.train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f'Starting training for {self.epochs} epochs...')

        self.build_ddp_model(cfg, device)
        self.device = device
        self.break_iter = -1
        self.break_epoch = -1
    def build_dataloader(self, cfg, callbacks):
        gs = max(int(self.model.stride.max()), 32)
        nl = self.model.head.nl
        self.imgsz = check_img_size(cfg.Dataset.img_size, gs, floor=gs * 2)
        print('self imgsz:', self.imgsz)

        if self.cuda and self.RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            self.model = torch.nn.DataParallel(self.model)

        if self.sync_bn and self.cuda and self.RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')
        self.train_loader, self.dataset = create_dataloader(self.data_dict['train'], self.imgsz, self.batch_size // self.WORLD_SIZE, gs, self.single_cls,
                                              hyp=cfg.hyp, augment=cfg.hyp.use_aug, cache=cfg.cache, rect=cfg.rect, rank=self.LOCAL_RANK,
                                              workers=cfg.Dataset.workers, prefix=colorstr('train: '),cfg=cfg)
        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())
        self.nb = len(self.train_loader)
        assert mlc < self.nc, f'Label class {mlc} exceeds nc={self.nc} in {cfg.Dataset.data_name}. Possible class labels are 0-{self.nc - 1}'

        if self.RANK in [-1, 0]:
            self.val_loader = create_dataloader(self.data_dict['val'] , self.imgsz, self.batch_size // self.WORLD_SIZE * 2, gs, self.single_cls,
                                       hyp=cfg.hyp, cache=None if self.noval else cfg.cache, rect=True, rank=-1,
                                       workers=cfg.Dataset.workers, pad=0.5,
                                       prefix=colorstr('val: '),cfg=cfg)[0]

            if not cfg.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                if self.plots:
                    plot_labels(labels, self.names, self.save_dir)

                if not cfg.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=cfg.hyp.anchor_t, imgsz=self.imgsz)
                self.model.half().float()

            callbacks.run('on_pretrain_routine_end')
        
        self.no_aug_epochs = cfg.hyp.no_aug_epochs

    def build_model(self, cfg, device):
        check_suffix(cfg.weights, '.pt')
        pretrained = cfg.weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(self.LOCAL_RANK):
                weights = attempt_download(cfg.weights)
            ckpt = torch.load(weights, map_location=device)
            self.model = Model(cfg or ckpt['model'].yaml).to(device)
            exclude = ['anchor'] if (cfg or cfg.Model.anchors) and not cfg.resume else []
            csd = ckpt['model'].float().state_dict()
            if cfg.prune_finetune:
                dynamic_load(self.model, csd,reinitialize=cfg.reinitial)
                if cfg.reinitial:
                    LOGGER.info("*** Reinitialize all")
                    self.model._initialize_biases()
                self.model.info()
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)
            self.model.load_state_dict(csd, strict=False)
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')
        else:
            self.model = Model(cfg).to(device)
            ckpt = None
        freeze = [f'model.{x}.' for x in range(cfg.freeze_layer_num)]
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
        
        self.ema = ModelEMA(self.model) if self.RANK in [-1, 0] else None

        self.start_epoch = 0
        pretrained = cfg.weights.endswith('.pt') and not cfg.reinitial
        if pretrained:
            if ckpt['optimizer'] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    LOGGER.info('pretrain model with different type of optimizer')

            if self.ema and ckpt.get('ema'):
                try:
                    self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                    self.ema.updates = ckpt['updates']
                except:
                    LOGGER.info('pretrain model with different type of ema')

            self.start_epoch = ckpt['epoch'] + 1
            if cfg.resume:
                assert self.start_epoch > 0, f'{weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
                self.epochs += ckpt['epoch']

        self.epoch = self.start_epoch
        self.model_type = self.model.model_type
        self.detect = self.model.head
        return ckpt

    def build_optimizer(self, cfg,optinit=True,weight_masks =None,ckpt=None):
        nbs = 64
        self.accumulate = max(round(nbs / self.batch_size), 1)
        weight_decay = cfg.hyp.weight_decay*self.batch_size * self.accumulate / nbs
        LOGGER.info(f"Scaled weight_decay = {weight_decay}")

        g_bnw, g_w, g_b = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g_b.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g_bnw.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g_w.append(v.weight)

        if not cfg.Model.RepOpt:
            if cfg.adam:
                self.optimizer = AdamW(g_b, lr=cfg.hyp.lr0,betas=(cfg.hyp.momentum, 0.999))
            else:
                self.optimizer = SGD(g_b, lr=cfg.hyp.lr0, momentum=cfg.hyp.momentum, nesterov=True)
            self.optimizer.add_param_group({'params': g_w, 'weight_decay': weight_decay})
            self.optimizer.add_param_group({'params': g_bnw})
        else:
            from models.optimizers.RepOptimizer import RepVGGOptimizer
            assert cfg.Model.RepScale_weight
            if self.opt_scales is None:
                scales = torch.load(cfg.Model.RepScale_weight, map_location=self.device)
            else:
                scales = self.opt_scales
            assert not cfg.adam, "RepOptimizer Only Support SGD."
            params_groups = [
                {'params': g_bnw},
                {'params': g_w, 'weight_decay': weight_decay},
                {'params': g_b}
            ]

            reinit = False
            if cfg.weights=='' and optinit:
                reinit = True

            self.optimizer = RepVGGOptimizer(self.model,scales,cfg,reinit=reinit,device=self.device,params=params_groups,weight_masks=weight_masks)
        LOGGER.info(f"{colorstr('optimizer:')} {type(self.optimizer).__name__} with parameter groups "
                f"{len(g_w)} weight, {len(g_bnw)} weight (no decay), {len(g_b)} bias")
        del g_w, g_bnw, g_b

        if cfg.linear_lr:
            self.lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - cfg.hyp.lrf) + cfg.hyp.lrf
        else:
            self.lf = one_cycle(1, cfg.hyp.lrf, self.epochs)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.scheduler.last_epoch = self.epoch - 1
        self.scaler = amp.GradScaler(enabled=self.cuda)
        if ckpt is not None and 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            print("Load Optimizer statedict")
            self.optimizer.load_state_dict(ckpt['optimizer'])

    def set_env(self, cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks):
        self.save_dir, self.epochs, self.batch_size, weights, self.single_cls, data, self.noval, self.nosave  = \
        Path(cfg.save_dir), cfg.epochs, cfg.Dataset.batch_size, cfg.weights, cfg.single_cls, cfg.Dataset.data_name, \
        cfg.noval, cfg.nosave
        self.sync_bn = cfg.sync_bn
        self.save_period = cfg.save_period
        self.device = device

        self.warmup_epochs = cfg.hyp.warmup_epochs
        self.momentum = cfg.hyp.momentum
        self.warmup_momentum = cfg.hyp.warmup_momentum
        self.warmup_bias_lr = cfg.hyp.warmup_bias_lr

        self.LOCAL_RANK = LOCAL_RANK
        self.RANK = RANK
        self.WORLD_SIZE = WORLD_SIZE
        self.norm_scale = cfg.Dataset.norm_scale

        w = self.save_dir / 'weights'
        w.mkdir(parents=True, exist_ok=True)
        self.last, self.best = w / 'last.pt', w / 'best.pt'

        with open(self.save_dir / 'opt.yaml', 'w') as f:
            with redirect_stdout(f): print(cfg.dump())

        if RANK in [-1, 0]:
            loggers = Loggers(self.save_dir, weights, cfg, LOGGER)
            if loggers.wandb:
                data_dict = loggers.wandb.data_dict
                if cfg.resume:
                    weights, epochs, hyp = cfg.weights, cfg.epochs, cfg.hyp
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

        self.plots = True
        self.cuda = device.type != 'cpu'
        init_seeds(1 + RANK)
        self.data_dict = {}
        self.data_dict['train'] = cfg.Dataset.train
        self.data_dict['val'] = cfg.Dataset.val
        self.data_dict['nc'] = cfg.Dataset.nc
        self.data_dict['names'] = cfg.Dataset.names
        self.nc = 1 if self.single_cls else int(self.data_dict['nc'])
        self.names = ['item'] if self.single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']
        assert len(self.names) == self.nc, f'{len(self.names)} names found for nc={self.nc} dataset in {data}'

    def build_ddp_model(self, cfg, device):
        if self.cuda and self.RANK != -1:
            self.model = DDP(self.model, device_ids=[self.LOCAL_RANK], output_device=self.LOCAL_RANK, find_unused_parameters=True)
      
        self.model.nc = self.nc
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(device) * self.nc
        self.model.names = self.names

        if cfg.Loss.type == 'ComputeLoss': 
            self.compute_loss = ComputeLoss(self.model, cfg)
        elif cfg.Loss.type == 'ComputeFastXLoss':
            self.compute_loss = ComputeFastXLoss(self.model, cfg)
        elif cfg.Loss.type == 'ComputeNanoLoss':
            self.compute_loss = ComputeNanoLoss(self.model, cfg)
        else:
            raise NotImplementedError

        is_distributed = is_parallel(self.model)
        if is_distributed:
            self.detect = self.model.module.head
        else:
            self.detect = self.model.head

    def before_train(self):
        return 0
    
    def build_train_logger(self):
        self.meter = MetricMeter()
        log_contents = ['Epoch', 'gpu_mem', 'labels', 'img_size']

        self.log_contents = log_contents
    
    def update_train_logger(self):
        for (imgs, targets, paths, _) in self.train_loader:
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
            if self.RANK in [-1, 0]:
                for loss_key in loss_items.keys():
                    self.log_contents.append(loss_key)
            break
        LOGGER.info(('\n' + '%10s' * len(self.log_contents)) % tuple(self.log_contents))
    
    def before_epoch(self):
        self.model.train()
        self.build_train_logger()
        self.update_train_logger()

        if self.epoch == self.epochs - self.no_aug_epochs:
            LOGGER.info("--->No mosaic aug now!")
            self.dataset.mosaic = False
            LOGGER.info("--->Add additional L1 loss now!")
            if self.model_type == 'yolox':
                self.detect.use_l1 = True

        self.meter = MetricMeter()

        if self.warmup_epochs > 0:
             self.nw = max(round(self.warmup_epochs * self.nb), 1000)
             self.nw = min(self.nw, (self.epochs - self.start_epoch) / 2 * self.nb)
        else:
             self.nw = -1

        if self.RANK != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
    
    def update_optimizer(self, loss, ni):
        self.scaler.scale(loss).backward()
                
        self.accumulate = max(round(64 / self.batch_size), 1) 

        if ni <= self.nw:
            xi = [0, self.nw]
            self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = ni
    
    def train_in_epoch(self, callbacks):

        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            if i == self.break_iter:
                break
            ni = i + self.nb * self.epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale

            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
                if self.RANK != -1:
                    loss *= self.WORLD_SIZE

            self.update_optimizer(loss, ni)

            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count= len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count+2)) % (
                        f'{self.epoch}/{self.epochs - 1}', mem, targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)
        self.lr = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()

    def after_epoch(self, callbacks, val):
        if self.RANK in [-1, 0]:
            callbacks.run('on_train_epoch_end', epoch=self.epoch)
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (self.epoch + 1 == self.epochs)
            if not self.noval:
                self.results, maps, _ = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=self.ema.ema,
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points = self.cfg.Dataset.np,
                                           val_kp=self.cfg.Dataset.val_kp)

            fi = fitness(np.array(self.results).reshape(1, -1))
            if fi > self.best_fitness:
                self.best_fitness = fi
            log_vals = list(self.meter.get_avg())[:3] + list(self.results) + self.lr
            callbacks.run('on_fit_epoch_end', log_vals, self.epoch, self.best_fitness, fi)

            if (not self.nosave) or (final_epoch):
                ckpt = {'epoch': self.epoch,
                        'best_fitness': self.best_fitness,
                        'model': deepcopy(de_parallel(self.model)).half(),
                        'ema': deepcopy(self.ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id':  None}

                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
                    w = self.save_dir / 'weights'
                    torch.save(ckpt, w / f'epoch{self.epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', self.last, self.epoch, final_epoch, self.best_fitness, fi)
    
    def after_train(self, callbacks, val):
        results = (0, 0, 0, 0, 0, 0, 0)
        if self.RANK in [-1, 0]:
            for f in self.last, self.best:
                if f.exists():
                    strip_optimizer(f)
                    if f is self.best:
                        LOGGER.info(f'\nValidating {f}...')
                        results, _, _ = val.run(self.data_dict,
                                            batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                            imgsz=self.imgsz,
                                            model=attempt_load(f, self.device).half(),
                                            iou_thres=0.65,
                                            single_cls=self.single_cls,
                                            dataloader=self.val_loader,
                                            save_dir=self.save_dir,
                                            save_json=False,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=self.compute_loss,
                                            num_points=self.cfg.Dataset.np,
                                            val_kp=self.cfg.Dataset.val_kp)

            callbacks.run('on_train_end', self.last, self.best, self.plots, self.epoch)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        torch.cuda.empty_cache()
        return results

    def train(self, callbacks, val):
        t0 = time.time()
        self.last_opt_step = -1
        self.results = (0, 0, 0, 0, 0, 0, 0)
        self.best_fitness = 0

        for self.epoch in range(self.start_epoch, self.epochs):
            if self.epoch == self.break_epoch:
                break
            self.before_epoch()
            self.train_in_epoch(callbacks)
            self.after_epoch(callbacks, val)
        results = self.after_train(callbacks, val)
        if self.RANK in [-1, 0]:
            LOGGER.info(f'\n{self.epoch - self.start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        return results