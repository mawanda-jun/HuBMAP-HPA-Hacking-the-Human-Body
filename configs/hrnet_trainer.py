# minibatches = 697  # Resize at 1536
# minibatches = 399  # resize at 1024
minibatches = 21  # resize at 6000

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=minibatches*5,
    warmup_ratio=1e-10,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=minibatches * 1200)
checkpoint_config = dict(by_epoch=False, interval=minibatches * 10)
evaluation = dict(interval=minibatches*2, metric='mDice', pre_eval=True)

gpu_ids = range(0, 1)
auto_resume = False

data = dict(
    samples_per_gpu=15,
    workers_per_gpu=16
)
