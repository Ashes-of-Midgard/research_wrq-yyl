"""Shell Script:
!python tools/train_det.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_sirst.py

python tools/train_det.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_sirst.py \
    --work-dir work_dirs/faster_rcnn_r50_fpn_1x_sirst_gpu_0 \
    --gpu-id 0

python tools/train_det.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_sirst.py \
    --work-dir work_dirs/faster_rcnn_r50_fpn_1x_sirst_gpu_1 \
    --gpu-id 1

python tools/train_det.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_sirst.py \
    --work-dir work_dirs/faster_rcnn_r50_fpn_1x_sirst_gpu_2 \
    --gpu-id 2

python tools/train_det.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_sirst.py \
    --work-dir work_dirs/faster_rcnn_r50_fpn_1x_sirst_gpu_3 \
    --gpu-id 3
"""

_base_ = [
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

sirst_version = 'sirstv2' # 'sirstv1' or 'sirstv2'
depth = 50
fpn_strides = [4, 8, 16, 32]
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

############################## dataset settings ##############################
dataset_type = 'SIRSTDet2NoCoDataset'
data_root = 'data/open-sirst-v2/'
img_norm_cfg = dict(
    mean=[111.89, 111.89, 111.89], std=[27.62, 27.62, 27.62], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
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
            noco_mode='noco_peak',
            ann_file=[data_root + split_cfg['train_split'],],
            img_prefix=[data_root,],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        noco_mode='noco_peak',
        ann_file=data_root + split_cfg['val_split'],
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        noco_mode='noco_peak',
        ann_file=data_root + split_cfg['test_split'],
        img_prefix=data_root,
        pipeline=test_pipeline)
)

############################## model settings ##############################
model = dict(
    type='FasterRCNN_AAL',
    visualization = True,
    visual_dir = 'work_dirs/faster_rcnn_r50_sirst_aal/visual',
    backbone=dict(
        type='ResNetSP',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        sp_attn_stem_out=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
checkpoint_config = dict(interval=1)