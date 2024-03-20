default_scope = 'opencd'

work_dir = 'work_dirs/lervicd/ttp_levit_large_levircd'

custom_imports = dict(imports=['mmseg.ttp'], allow_failed_imports=False)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=10),
    dist_cfg=dict(backend='nccl'),
)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10, save_best='cd/iou_changed', max_keep_ckpts=5, greater_keys=['cd/iou_changed'], save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1,
                       img_shape=(1024, 1024, 3))
)
vis_backends = [dict(type='CDLocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='samcd', group='levircd', name='ttp_sam_large_levircd'))
                ]

visualizer = dict(
    type='CDLocalVisualizer',
    vis_backends=vis_backends, name='visualizer', alpha=1.0)
log_processor = dict(by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

crop_size = (512, 512)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32)
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
fpn_norm_cfg = dict(type='LN2d', requires_grad=True)

pretrain_ckpt_path = 'https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth'

model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MMPretrainSamVisionEncoder',
        encoder_cfg=dict(
            type='mmpretrain.PyramidVisionTransformer',
            arch='custom',
            img_size=crop_size[0],
            patch_size=16,
            drop_path_rate=0,
            attn_drop_rate=0,
            mlp_ratios=None,
            out_indices=(0, ),
            //init_cfg=dict(type='Pretrained', checkpoint=pretrain_ckpt_path, prefix='backbone.'),
        ),
    ),
    neck=dict(
        type='SequentialNeck',
        necks=[
            dict(
                type='FeatureFusionNeck',
                policy='concat',
                out_indices=(0,)),
            dict(
                type='SimpleFPN',
                backbone_channel=512,
                in_channels=[128, 256, 512, 512],
                out_channels=256,
                num_outs=5,
                norm_cfg=fpn_norm_cfg),
        ],
    ),
    decode_head=dict(
        type='MLPSegHead',
        out_size=(128, 128),
        in_channels=[256]*5,
        in_index=[0, 1, 2, 3, 4],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2))
)  # yapf: disable

dataset_type = 'LEVIR_CD_Dataset'
data_root = '/mnt/levir_datasets/levir-cd'


train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

batch_size_per_gpu = 32

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='train/label',
            img_path_from='train/A',
            img_path_to='train/B'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='test/label',
            img_path_from='test/A',
            img_path_to='test/B'),
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CDMetric',
)
test_evaluator = val_evaluator

max_epochs = 300
base_lr = 0.0004
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-4, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        begin=5,
        by_epoch=True,
        end=max_epochs,
        convert_to_iter_based=True
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
)

