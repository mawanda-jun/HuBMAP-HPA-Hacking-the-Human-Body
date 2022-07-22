_base_ = ['./dataset_512.py', './efficientnet_trainer.py']

# model settings
backbone_checkpoint = "/home/mawanda/projects/HuBMAP/.cache/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth"
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint=backbone_checkpoint)),
    decode_head=dict(
        type='UPerHead',
        in_channels=[48, 136, 384],
        in_index=[0, 1, 2],
        pool_scales=(2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='LovaszLoss',
            loss_type="multi_class",
            loss_weight=1.0,
            reduction="none")
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='LovaszLoss',
            loss_type="multi_class",
            loss_weight=0.4,
            reduction="none")
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# fp16 settings
optimizer_config = dict(
    type='Fp16OptimizerHook', 
    loss_scale=512.,
    grad_clip=None
)
# fp16 placeholder
fp16 = dict()

# Load weights from checkpoint
# load_from = '/home/mawanda/projects/HuBMAP/.cache/upernet_r50_512x512_80k_ade20k_20200614_144127-ecc8377b.pth'
load_from = backbone_checkpoint
