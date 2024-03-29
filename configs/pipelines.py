crop_size = (512, 512)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True)

albu_color_transforms = [
    dict(type="HorizontalFlip", p=.25),
    dict(type="VerticalFlip", p=0.25),
    dict(type="RandomRotate90", p=0.25),
    dict(type="Transpose", p=0.25),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.3),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    dict(type="Resize", height=crop_size[0], width=crop_size[1], always_apply=True),
    # dict(
    #     type="PadIfNeeded",
    #     min_height=crop_size[0],
    #     min_width=crop_size[1], 
    #     p=1
    # )
]

# Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type="RandomMosaic", prob=0.5, img_scale=(1024, 1024)),
    # dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomRotate', prob=0.5, degree=(0, 180)),
    dict(type='RandomFlip', prob=0., direction='horizontal'),  # Needed for metadata "flip" key
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
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', img_scale=(2048, 1024)),
#     # dict(type="RandomMosaic", prob=0.5, img_scale=(2048, 2048)),
#     # dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
#     # dict(type='RandomRotate', prob=0.5, degree=(0, 180)),
#     # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     # dict(type='RandomFlip', prob=0.5, direction='vertical'),
#     # dict(type="RandomCutOut", prob=0.5, n_holes=(1, 10), cutout_shape=(0, 3)),
#     # dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     # dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
#     # dict(type='ImageToTensor', keys=['img']),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img'])
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=crop_size,
        img_ratios=[.5, 1., 1.5],
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