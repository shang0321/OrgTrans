from .yacs import CfgNode as CN

_C = CN()
_C.debug= False 
_C.do_test=False
_C.finetune=False
_C.device= ''
_C.ngpu=1
_C.adam=False
_C.prune_finetune=False
_C.reinitial=False
_C.noautoanchor=True 
_C.project=''
_C.name='exp'
_C.epochs=300
_C.val_conf_thres = 0.001
_C.local_rank=-1
_C.save_period=-1
_C.weights=''
_C.freeze_layer_num = 0
_C.cache=False
_C.rect=False
_C.save_dir=''
_C.single_cls=False
_C.evolve=False
_C.noval=False
_C.nosave=False
_C.sync_bn=False
_C.resume=False
_C.exist_ok=False
_C.linear_lr=False
_C.check_datacache=False
_C.entity=None
_C.upload_dataset=False
_C.bbox_interval=-1
_C.artifact_alias='latest'
_C.find_unused_parameters=False

_C.hyp=CN()
_C.hyp.use_aug=True
_C.hyp.lr0=0.01
_C.hyp.lrf=0.01
_C.hyp.momentum=0.937
_C.hyp.weight_decay=0.0005
_C.hyp.warmup_epochs=0
_C.hyp.warmup_momentum=0.8
_C.hyp.warmup_bias_lr=0.1

_C.hyp.hsv_h=0.5
_C.hyp.hsv_s=0.5
_C.hyp.hsv_v=0.5
_C.hyp.degrees=0.0
_C.hyp.translate=0.1
_C.hyp.scale=0.5
_C.hyp.shear=0.0
_C.hyp.perspective=0.0
_C.hyp.flipud=0.0
_C.hyp.fliplr=0.5
_C.hyp.mosaic=1.0
_C.hyp.mixup=0.0
_C.hyp.burn_epochs=1
_C.hyp.copy_paste=0.0
_C.hyp.no_aug_epochs=0
_C.hyp.cutout=0.0 

_C.Model=CN()
_C.Model.weights=''

_C.Model.width_multiple = 1.0
_C.Model.depth_multiple = 1.0
_C.Model.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
_C.Model.ch = 3

_C.Model.Backbone = CN()
_C.Model.Backbone.name = 'darknet'
_C.Model.Backbone.stage_repeats = [4, 8, 4]
_C.Model.Backbone.output_layers = [6, 14, 18]
_C.Model.Backbone.model_size = '0.2x'
_C.Model.Backbone.activation = 'LeakyReLU'
_C.Model.Backbone.arch = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1] 
_C.Model.Backbone.first_input_channels = 1
_C.Model.Backbone.out_stages = [2, 3, 4]
_C.Model.Backbone.kernel_size = 3
_C.Model.Backbone.with_last_conv = False
_C.Model.Backbone.pretrain = False
_C.Model.Backbone.in_channels = 3
_C.Model.Backbone.out_channels = [64, 128, 256, 512, 1024]
_C.Model.Backbone.num_repeats = [1, 6, 12, 18, 6]
_C.Model.Backbone.lite_conv = False

_C.Model.Neck = CN()
_C.Model.Neck.name = 'darknet'
_C.Model.Neck.in_channels = [32, 64, 128]
_C.Model.Neck.out_channels = [32]
_C.Model.Neck.start_level = 0
_C.Model.Neck.end_level = -1
_C.Model.Neck.num_outs = 3
_C.Model.Neck.activation = 'ReLU'
_C.Model.Neck.interpolate = 'bilinear'
_C.Model.Neck.num_repeats = [12, 12, 12, 12]

_C.Model.Head = CN()
_C.Model.Head.name = 'darknet'
_C.Model.Head.share_cls_reg = False
_C.Model.Head.activation = 'SiLU'
_C.Model.Head.conv_type = 'DWConv'
_C.Model.Head.stacked_convs= 2
_C.Model.Head.octave_base_scale= 5
_C.Model.Head.feat_channels = 256
_C.Model.Head.strides = [8,16,32]
_C.Model.Head.in_channels = [128, 256, 512]
_C.Model.Head.num_decouple = 2

_C.Model.RepOpt = False
_C.Model.RepScale_weight = ''
_C.Model.RealVGGModel = False
_C.Model.LinearAddModel = False
_C.Model.QARepVGGModel = False

_C.Model.inplace=True
_C.Model.prior_prob = 0.01

_C.Loss = CN()

_C.Loss.type = 'ComputeXLoss'
_C.Loss.box=0.05
_C.Loss.cls=0.5
_C.Loss.cls_pw=1.0
_C.Loss.obj=1.0
_C.Loss.obj_pw=1.0
_C.Loss.fl_gamma=0.0
_C.Loss.autobalance= False
_C.Loss.label_smoothing=0.0
_C.Loss.anchor_t = 4.0
_C.Loss.kp_loss_weight = 10.0
_C.Loss.static_assigner_epoch = 5
_C.Loss.single_targets=False

_C.Loss.qfl_use_sigmoid=True 
_C.Loss.qfl_beta=2.0
_C.Loss.qfl_loss_weight=1.0
_C.Loss.dfl_loss_weight=0.25
_C.Loss.reg_max=7

_C.Loss.box_loss_weight=5.0
_C.Loss.obj_loss_weight=1.0
_C.Loss.cls_loss_weight=1.0
_C.Loss.iou_obj=False

_C.Loss.use_dfl=True
_C.Loss.grid_cell_size=5.0
_C.Loss.grid_cell_offset=0.5
_C.Loss.iou_type='giou'
_C.Loss.use_gfl=False

_C.Loss.top_k=13
_C.Loss.assigner_type='TAL'
_C.Loss.embedding=64

_C.Dataset=CN()
_C.Dataset.train=''
_C.Dataset.val=''
_C.Dataset.test=''
_C.Dataset.target = ''
_C.Dataset.img_path=''
_C.Dataset.label_path=''

_C.Dataset.batch_size=96
_C.Dataset.img_size=640
_C.Dataset.rect=False
_C.Dataset.workers=16
_C.Dataset.quad=False
_C.Dataset.nc=80
_C.Dataset.np=0
_C.Dataset.num_ids=0
_C.Dataset.pseudo_ids=False
_C.Dataset.names=[]
_C.Dataset.include_class=[]
_C.Dataset.data_name='default_name'
_C.Dataset.sampler_type='normal'
_C.Dataset.norm_scale=255.0
_C.Dataset.debug= False
_C.Dataset.val_kp= False

_C.Qat=CN()
_C.Qat.use_qat = False
_C.Qat.quant_dir = False
_C.Qat.bitmode = 'int8'
_C.Qat.backend = 'tensorrt'
_C.Qat.use_defaultfuse = False  
_C.Qat.use_quant_sensitivity_analysis = True
_C.Qat.sensitive_num = -1
_C.Qat.sensitive_relerror = 0.01
_C.Qat.sensitive_eval_batch = 30

_C.Prune=CN()
_C.Prune.use_sparse = False
_C.Prune.sparse_rate = 1e-3
_C.Prune.flops_target = 0.3
_C.Prune.prune_freq = 50
_C.Prune.channel_divide = 8
_C.Prune.iterative_prune = False

_C.Prune.ft_reinit = False
_C.Prune.prune_finetune=False
_C.Prune.sr_type = ''
_C.Prune.update_sr = False

_C.Distill=CN()
_C.Distill.use_distill=False
_C.Distill.dist_loss='l2'
_C.Distill.Tmodel=''
_C.Distill.temp=20
_C.Distill.giou=0.05
_C.Distill.dist=1.0
_C.Distill.boxloss=False
_C.Distill.objloss=False
_C.Distill.clsloss=False
_C.Distill.loss_type=''

_C.SSOD = CN()
_C.SSOD.train_domain = True
_C.SSOD.extra_teachers = [ ]
_C.SSOD.extra_teachers_class_names = [ ]
_C.SSOD.conf_thres = 0.65
_C.SSOD.valid_thres = 0.55
_C.SSOD.nms_conf_thres = 0.1
_C.SSOD.nms_iou_thres = 0.65
_C.SSOD.teacher_loss_weight = 0.5
_C.SSOD.cls_loss_weight = 0.5
_C.SSOD.box_loss_weight = 0.1
_C.SSOD.obj_loss_weight = 0.5
_C.SSOD.focal_loss= 0.0
_C.SSOD.loss_type = 'ComputeStudentMatchLoss'
_C.SSOD.pseudo_label_type='FairPseudoLabel'
_C.SSOD.debug=False
_C.SSOD.with_da_loss = False
_C.SSOD.da_loss_weights = 0.01
_C.SSOD.ema_rate= 0.999
_C.SSOD.ignore_thres_high=0.7
_C.SSOD.ignore_thres_low=0.2
_C.SSOD.dynamic_thres_epoch=0
_C.SSOD.uncertain_aug=True
_C.SSOD.use_ota= False
_C.SSOD.multi_label= False
_C.SSOD.ignore_obj = False
_C.SSOD.resample_high_percent=0.3
_C.SSOD.resample_low_percent=0.95
_C.SSOD.multi_step_lr=False
_C.SSOD.milestones=[10, 20]
_C.SSOD.pseudo_label_with_obj=True
_C.SSOD.pseudo_label_with_bbox=True
_C.SSOD.pseudo_label_with_cls=False
_C.SSOD.epoch_adaptor=True
_C.SSOD.teacher_ota_cost=False
_C.SSOD.iou_type='giou'
_C.SSOD.cosine_ema=True
_C.SSOD.imitate_teacher=False
_C.SSOD.fixed_accumulate=False
_C.SSOD.use_soft_nms=False
_C.SSOD.soft_nms_sigma=0.5
_C.SSOD.nms_time_limit=60.0

_C.SSOD.dynamic_thres=False
_C.SSOD.dynamic_thres_factor=1.5
_C.SSOD.min_conf_thresh=0.1

_C.SSOD.uncertainty_thresh=0.3

_C.SSOD.quality_weight=False
_C.SSOD.quality_weight_type='gaussian'
_C.SSOD.quality_weight_factor=2.0

_C.SSOD.progressive_blend = True
_C.SSOD.blend_start_epoch = 10
_C.SSOD.blend_end_epoch = 100
_C.SSOD.blend_warmup_epochs = 5

_C.SSOD.curriculum_distillation = False
_C.SSOD.confidence_threshold_start = 0.8
_C.SSOD.confidence_threshold_end = 0.3
_C.SSOD.curriculum_start_epoch = 0
_C.SSOD.curriculum_end_epoch = 300
_C.SSOD.curriculum_strategy = 'linear'

_C.SSOD.ssod_hyp = CN()
_C.SSOD.ssod_hyp.mosaic=1.0
_C.SSOD.ssod_hyp.degrees=0.0
_C.SSOD.ssod_hyp.translate=0.1
_C.SSOD.ssod_hyp.scale=0.5
_C.SSOD.ssod_hyp.shear=0.0
_C.SSOD.ssod_hyp.flipud=0.0
_C.SSOD.ssod_hyp.fliplr=0.5
_C.SSOD.ssod_hyp.perspective=0.0
_C.SSOD.ssod_hyp.hsv_h=0.015
_C.SSOD.ssod_hyp.hsv_s=0.7
_C.SSOD.ssod_hyp.hsv_v=0.4
_C.SSOD.ssod_hyp.with_gt=False
_C.SSOD.ssod_hyp.cutout=0.9
_C.SSOD.ssod_hyp.autoaugment=0.9

_C.NAS = CN()
_C.NAS.use_nas = False
_C.NAS.width_range = []
_C.NAS.params_target = [0,1e10]
_C.NAS.flops_target = [0,1e10]
_C.NAS.GEA = CN()
_C.NAS.GEA.pop_size = 10
_C.NAS.GEA.sample_size = 3
_C.NAS.GEA.sample_epochs = 20
_C.NAS.GEA.sample_dataIter = -1
_C.NAS.GEA.cycles = 100

_C.SSOD.use_moco = False
_C.SSOD.moco_dim = 256
_C.SSOD.moco_k = 65536
_C.SSOD.moco_m = 0.999
_C.SSOD.moco_t = 0.2
_C.SSOD.moco_loss_weight = 0.5
_C.SSOD.use_wavelet = False

def get_cfg():
    return _C.clone()