# RTMPose-to-3D single-image pose lifting

This is the second, deployment-oriented training route. It fine-tunes
MMPose's `ImagePoseLifter + TCN` with one frame of estimated 2D joints:

`H36M JPEG -> RTMDet-M -> RTMPose-M -> COCO-to-H36M 17 joints -> TCN -> root-relative 3D pose`.

The 3D labels are Human3.6M camera-coordinate joints. Subjects S1, S5, S6,
S7 and S8 form the training split; S9 and S11 are held out for validation.

## Required local inputs

Obtain Human3.6M under its license. This route requires:

- `data/h36m/annotation_body3d/fps10/h36m_{train,test}_keypoints.npz`, which
  pair H36M 3D labels with image sample identifiers;
- `data/h36m_raw/archives/S*.tar`, containing the source JPEGs;
- local RTMDet-M and RTMPose-M checkpoints.

None of those datasets or model weights are committed to Git.

To convert the compact paired annotation archive without extracting images:

```bash
python tools/dataset_converters/convert_h36m_annot_h5.py \
  --archive data/h36m/h36m_annot.tar --output-root data/h36m
```

## Generate detector-matched 2D joints

Run once for the training subjects, then once for the held-out subjects. The
generator streams JPEGs from the subject archives, keeps output in exact NPZ
row order, and can resume after a completed subject.

```bash
python projects/single_image_pose_lift/generate_h36m_rtmpose_from_tar.py \
  --ann-file data/h36m/annotation_body3d/fps10/h36m_train_keypoints.npz \
  --archive-dir data/h36m_raw/archives \
  --det-config demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  --det-checkpoint /path/to/rtmdet_m.pth \
  --pose-config configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
  --pose-checkpoint /path/to/rtmpose-m.pth \
  --output data/h36m/annotation_body3d/fps10/rtmdet_rtmpose_m_train_h36m17.npy \
  --subjects S1 S5 S6 S7 S8 --bbox-thr 0.3 --batch-size 16 --device cuda:0
```

For validation, replace `train` by `test` in both paths and pass
`--subjects S9 S11`. Add `--resume` to continue a previously interrupted
generation. Generated arrays are intentionally ignored by Git.

## Train

```bash
python tools/train.py \
  projects/single_image_pose_lift/image_pose_lift_tcn_h36m_rtmpose_v2.py \
  --work-dir work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2
```

The configuration trains for 80 epochs with Adam (`lr=1e-3`), batch size 512,
and `MultiStepLR` decay at epochs 50 and 70 (`1e-3 -> 1e-4 -> 1e-5`). It
accepts a single 17-joint pose per sample (`seq_len=1`); it is not a temporal
model. Training checkpoints and logs remain under `work_dirs/` and are not
committed.

## Evaluate a paired sample

`demo_h36m_rtmpose_lift.py` produces an image with the RTMPose 2D overlay and
side-by-side predicted/ground-truth 3D skeletons when given a held-out H36M
image and its NPZ row index. `benchmark_s1_rtmpose_lift.py` applies the same
path to selected archive samples.

This model predicts root-relative pose, not calibrated metric world position;
single-image depth remains inherently ambiguous.
