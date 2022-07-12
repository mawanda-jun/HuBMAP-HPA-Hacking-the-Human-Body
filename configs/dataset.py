crop_size = (1024, 1024)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True)

# Pipelines
train_pipeline = [
    dict(type="RandomMosaic", prob=0.5, img_scale=(2048, 2048)),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomRotate', prob=0.5, degree=(0, 180)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type="RandomCutOut", prob=0.5, n_holes=(1, 10), cutout_shape=(0, 3)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
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
        img_scale=(2048, 1024),
        img_ratios=[1., 1.0],
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
data_root = '/home/mawanda/Documents/HuBMAP/for_mmdetection/'
classes = ('organ', )

data = dict(
    samples_per_gpu=7,
    workers_per_gpu=8,
    # train=dict(
    #     type="MultiImageMixDataset",
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
                dict(type='LoadAnnotations')
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
        pipeline=test_pipeline
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
