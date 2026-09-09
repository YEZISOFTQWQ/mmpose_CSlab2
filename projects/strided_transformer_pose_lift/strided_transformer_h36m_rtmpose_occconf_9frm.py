"""v4 preparation: confidence- and occlusion-aware temporal 2D-to-3D lifting.

This is intentionally a separate experiment from v3. It has a different input
dimension (17 x [x, y, confidence, mask]) and must never overwrite v3 outputs.
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
    image_size=(1000, 1002),
    root_index=0,
    remove_root=True,
    save_index=True,
    concat_vis=True,
    concat_mask=True)

model = dict(
    type='PoseLifter',
    backbone=dict(
        type='StridedTransformerBackbone',
        in_channels=17 * 4,
        keypoint_dim=4,
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
    # Input-only corruption: labels remain original H36M 3D poses.
    dict(type='TemporalJointOcclusion', probability=.75,
         span_lengths=(1, 2, 4), groups_per_window=1),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=(
        'id', 'category_id', 'target_img_path', 'flip_indices', 'target_root',
        'target_root_index'))
]
val_pipeline = [
    # No synthetic corruption at validation: mask is automatically all ones.
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=(
        'id', 'category_id', 'target_img_path', 'flip_indices', 'target_root',
        'target_root_index'))
]

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
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
    num_workers=4,
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

work_dir = 'work_dirs/strided_transformer_h36m_rtmpose_occconf_9frm'
