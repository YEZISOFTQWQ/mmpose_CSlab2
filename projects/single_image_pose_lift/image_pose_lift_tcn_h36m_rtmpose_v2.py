"""Single-frame H36M 2D-to-3D training using deployment-style RTMPose input.

The annotations remain paired H36M 2D/3D samples, but ``keypoint_2d_src`` is
set to ``detection`` so the input comes from RTMDet + RTMPose predictions.
S1/S5/S6/S7/S8 train; S9/S11 are held out by the two NPZ splits.
"""

_base_ = [
    '../../configs/body_3d_keypoint/image_pose_lift/h36m/'
    'image-pose-lift_tcn_8xb64-200e_h36m.py'
]

train_cfg = dict(max_epochs=80, val_interval=5)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[50, 70], gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=512)
work_dir = 'work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2'
default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

train_dataloader = dict(
    batch_size=512,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file='annotation_body3d/fps10/h36m_train_keypoints.npz',
        seq_len=1,
        causal=True,
        keypoint_2d_src='detection',
        keypoint_2d_det_file='annotation_body3d/fps10/'
        'rtmdet_rtmpose_m_train_h36m17.npy',
        data_prefix=dict(img='')))

val_dataloader = dict(
    batch_size=512,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file='annotation_body3d/fps10/h36m_test_keypoints.npz',
        seq_len=1,
        causal=True,
        keypoint_2d_src='detection',
        keypoint_2d_det_file='annotation_body3d/fps10/'
        'rtmdet_rtmpose_m_test_h36m17.npy',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader
