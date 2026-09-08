"""Nine-frame RTMPose-to-3D training recipe for Human3.6M.

This is deliberately an offline experiment configuration. It leaves the
single-frame v2 model and its work directory untouched.
"""

custom_imports = dict(
    imports=['projects.strided_transformer_pose_lift'], allow_failed_imports=False)

_base_ = ['../../configs/_base_/default_runtime.py']

data_root = 'data/h36m/'
dataset_type = 'LazyHuman36mDataset'
sequence_length = 9

codec = dict(
    type='TemporalImagePoseLifting',
    num_keypoints=17,
    # H36M frame resolution. Inference must normalize each source frame by
    # the same width/height rule, not by the old single-frame statistics.
    image_size=(1000, 1002),
    root_index=0,
    remove_root=True,
    save_index=True)

model = dict(
    type='PoseLifter',
    backbone=dict(
        type='StridedTransformerBackbone',
        in_channels=17 * 2,
        seq_len=sequence_length,
        embed_dim=256,
        num_heads=8,
        feedforward_dim=512,
        num_vte_layers=3,
        strides=(3, 3),
        dropout=0.25),
    head=dict(
        type='FullToSingleRegressionHead',
        in_channels=256,
        num_joints=16,
        loss=dict(type='MPJPELoss', use_target_weight=True),
        sequence_loss=dict(type='MPJPELoss', use_target_weight=True),
        sequence_loss_weight=1.0,
        decoder=codec),
    test_cfg=dict(flip_test=False))

train_cfg = dict(max_epochs=80, val_interval=5)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4))
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[55, 70], gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=5, save_best='MPJPE', rule='less',
        max_keep_ckpts=3))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

train_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=(
        'id', 'category_id', 'target_img_path', 'flip_indices', 'target_root',
        'target_root_index'))
]
val_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='annotation_body3d/fps10/h36m_train_keypoints.npz',
        seq_len=sequence_length,
        seq_step=1,
        causal=False,
        pad_video_seq=False,
        keypoint_2d_src='detection',
        keypoint_2d_det_file=(
            'annotation_body3d/fps10/rtmdet_rtmpose_m_train_h36m17.npy'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='annotation_body3d/fps10/h36m_test_keypoints.npz',
        seq_len=sequence_length,
        seq_step=1,
        causal=False,
        pad_video_seq=True,
        keypoint_2d_src='detection',
        keypoint_2d_det_file=(
            'annotation_body3d/fps10/rtmdet_rtmpose_m_test_h36m17.npy'),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MPJPE', mode='mpjpe'),
    dict(type='MPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator

# A separate path protects the existing v2 result:
# work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/best_MPJPE_epoch_75.pth
work_dir = 'work_dirs/strided_transformer_h36m_rtmpose_9frm'
