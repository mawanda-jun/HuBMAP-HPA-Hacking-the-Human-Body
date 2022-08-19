_base_ = ['../segformer_mit-b5_512x512_160k_ade20k.py']

crop_size = (512, 512)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True)

albu_color_transforms = [
    dict(type="HorizontalFlip", p=.25),
    dict(type="VerticalFlip", p=0.25),
    dict(type="RandomRotate90", p=0.25),
    dict(type="Transpose", p=0.25),
    dict(type="RandomResizedCrop", height=crop_size[0], width=crop_size[1], scale=(0.8, 1.0), p=1.),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.2,
        scale_limit=0.0,
        rotate_limit=20,
        interpolation=1,
        p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    # dict(type="RandomGamma", p=0.2),
    # dict(type="CLAHE", p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=20,
    #             g_shift_limit=20,
    #             b_shift_limit=20,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.3),
    # dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #     ],
    #     p=0.1),
    # dict(type='PadIfNeeded', min_height=512, min_width=512, p=1)
    # dict(type="OneOf",
    #     transforms=[
    # dict(type="ElasticTransform", p=.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        dict(type="GridDistortion", p=.5),
    #     dict(type="OpticalDistortion", distort_limit=1, shift_limit=0.5, p=1),
    # ], p=0.3),
]

# Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type="RandomMosaic", prob=0.5, img_scale=(1024, 1024)),
    # dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomRotate', prob=0.5, degree=(0, 180)),
    # Needed for metadata "flip" key
    dict(type='RandomFlip', prob=0., direction='horizontal'),
    # dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # dict(type="RandomCutOut", prob=0.5, n_holes=(1, 10), cutout_shape=(0, 3)),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Albu',
        transforms=albu_color_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask'
        },
        update_pad_shape=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(1., 1.)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=crop_size,
        img_ratios=[1.],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Define datasets
dataset_type = 'CustomDataset'
data_root = '/home/mawanda/Documents/HuBMAP/for_mmdetection_resized_5000_inverted'
# data_root = '/home/mawanda/Documents/HuBMAP/for_mmdetection_multires_512'
# data_root = '/home/mawanda/Documents/HuBMAP/for_mmdetection_512'
classes = ('organ', )

data = dict(
    # samples_per_gpu=30,
    # workers_per_gpu=12,
    # train=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     classes=classes,
    #     img_dir='img_dir/train',
    #     ann_dir='ann_dir/train',
    #     img_suffix=".png",
    #     seg_map_suffix=".png",
    #     pipeline=train_pipeline
    # ),
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            classes=classes,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            img_suffix=".png",
            seg_map_suffix=".png",
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
            ],
        ),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline
    )
)
