input_size = 512
############### model settings ##################
model = dict(
    type='SingleStageDetectorSP',
    backbone=dict(
        type='ResNetSP_ExtraLayers',
        depth=34,
        input_size=input_size,
        strides=(1, 2, 2, 1),
        out_indices=(2, 3),
        frozen_stages=1,
        use_sp_attn_indices=(2, 3),
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet34')),
    neck=None,
    bbox_head=dict(
        type='SSDHeadAAL',
        in_channels=(256, 512, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            strides=(16, 16, 32, 64, 128, 256, 512),
            ratios=([2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]),
            basesize_ratio_range=(0.15, 0.9)),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            clip_border=True,
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        num_classes=1))
############ model training and testing settings ###################
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
######################## dataset setting #######################
sirst_version = 'sirstv2' # 'sirstv1' or 'sirstv2'
if sirst_version == 'sirstv1':
    split_cfg = {
        'train_split': 'splits/trainval_v1.txt',
        'val_split': 'splits/test_v1.txt',
        'test_split': 'splits/test_v1.txt',
    }
elif sirst_version == 'sirstv2':
    split_cfg = {
        'train_split': 'splits/trainval_full.txt',
        'val_split': 'splits/test_full.txt',
        'test_split': 'splits/test_full.txt',
    }
else:
    raise ValueError("wrong sirst_version")
dataset_type = 'SIRSTDet2NoCoDataset'
data_root = './data/open-sirst-v2/'
img_norm_cfg = dict(
    mean=[111.89, 111.89, 111.89], std=[27.62, 27.62, 27.62], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])])]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[data_root + 'splits/trainval_full.txt',],
            img_prefix=[data_root,],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'splits/test_full.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'splits/test_full.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='mAP')
########################## learning policy ###################################
optimizer = dict(type='SGD', lr=0.0003, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35,norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[16, 20])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssd512_r34_voc'
load_from = None
resume_from = None
workflow = [('train', 1)]
