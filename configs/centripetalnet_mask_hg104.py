#model settings
model = dict(
    type='CentripetalNet',
    backbone=dict(
        type='Hourglass',
        n=5,
        nstack=2,
        dims=[256, 256, 384, 384, 384, 512],
        modules=[2, 2, 2, 2, 2, 4],
        out_dim=80,),
    neck=None,
    bbox_head=dict(
        type='Centripetal_mask',
        num_classes=81,
        in_channels=256,
        with_mask=True,
        ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/mscoco2017/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)

cornernet_mode = True
    
data = dict(
    imgs_per_gpu=6,#3
    workers_per_gpu=3,#3
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=(511, 511),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False,
        cornernet_mode=cornernet_mode,
        extra_aug=dict(
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            expand=dict(
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1, 4)),
            random_crop=dict(
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3))),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(511, 511),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=1,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        cornernet_mode=cornernet_mode,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        img_prefix=data_root + 'test2017/',
        img_scale=(511, 511),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=1,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
        cornernet_mode=cornernet_mode,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='Adam', lr=0.00005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# policy='fixed'
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[190])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 210
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centripetalnet_mask_hg104'
resume_from = None
load_from = None
workflow = [('train', 1)]

