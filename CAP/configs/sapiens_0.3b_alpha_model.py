model_name = 'sapiens_0.3b'
embed_dim = 1024
input_height = 512
input_width = 512
input_size = (input_height, input_width)

custom_imports = dict(
    imports=[
        'coarse_alpha.head',
        'coarse_alpha.segmentor',
        'coarse_alpha.loss',
    ],
    allow_failed_imports=False,
)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=input_size,
    seg_pad_val=255,
)

model = dict(
    type='CoarseAlphaEstimator',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch=model_name,
        img_size=input_size,
        patch_size=16,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
    ),
    decode_head=dict(
        type='ViTCoarseAlphaHead',
        in_channels=embed_dim,
        channels=768,
        deconv_out_channels=(768, 768, 768),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=(768, 768, 768),
        conv_kernel_sizes=(1, 1, 1),
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='CoarseAlphaLoss',
            reduction='mean',
            loss_weight=1.0,
            laplacian_weight=1.0,
            max_levels=5,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
