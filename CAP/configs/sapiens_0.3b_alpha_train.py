_base_ = ['./sapiens_0.3b_alpha_model.py']

_os = __import__('os')

pretrained_checkpoint_name = 'sapiens_0.3b_epoch_1600_clean.pth'
pretrained_checkpoint = _os.environ.get('SAPIENS_PRETRAINED_CHECKPOINT', '')
if _os.path.basename(pretrained_checkpoint) != pretrained_checkpoint_name:
    raise RuntimeError(
        'SAPIENS_PRETRAINED_CHECKPOINT must point to '
        f'{pretrained_checkpoint_name}')

data_root = _os.environ.get('SAPIENS_ALPHA_DATA_ROOT', 'data')
training_roots = [
    _os.path.join(data_root, 'HHM50K_img_alpha'),
    _os.path.join(data_root, 'RVM_img_alpha'),
    _os.path.join(data_root, 'MatteHuman_img_alpha'),
]
training_repeat_factors = [2, 1, 2]
validation_roots = [
    _os.environ.get(
        'SAPIENS_ALPHA_VAL_ROOT',
        _os.path.join(data_root, 'validation_img_alpha'),
    )
]
del _os

input_height = 512
input_width = 512
input_size = (input_height, input_width)
pre_crop_height = 560
pre_crop_width = 560
pre_crop_size = (pre_crop_height, pre_crop_width)
num_layers = 24
num_epochs = 30

custom_imports = dict(
    imports=[
        'coarse_alpha.dataset',
        'coarse_alpha.transforms',
        'coarse_alpha.head',
        'coarse_alpha.segmentor',
        'coarse_alpha.loss',
        'coarse_alpha.metric',
        'mmseg.engine.optimizers.layer_decay_optim_wrapper',
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_checkpoint,
        ),
    ),
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
log_processor = dict(by_epoch=True)
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.85,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=400,
        by_epoch=False,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=num_epochs,
        by_epoch=True,
    ),
]

train_pipeline = [
    dict(type='ResizeImageAndAlpha', size=pre_crop_size),
    dict(type='RandomCropAndFlipPair', crop_size=input_size),
    dict(type='PackAlphaMattingInputs'),
]
val_pipeline = [
    dict(type='ResizeImageAndAlpha', size=input_size),
    dict(type='PackAlphaMattingInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ImageAlphaDataset',
        roots=training_roots,
        repeat_factors=training_repeat_factors,
        serialize_data=False,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ImageAlphaDataset',
        roots=validation_roots,
        serialize_data=False,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='AlphaMattingMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=num_epochs,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
