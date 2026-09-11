# MMPose batch inference desktop application

This desktop GUI has three selectable project models:

- **v2 single-frame**: `image/video frame -> RTMDet-M -> RTMPose-M -> H36M-17 -> v2 lifter`.
  It supports images, videos, and up to four independently inferred people.
- **v3 temporal**: `9 video frames -> RTMDet-M + RTMPose-M -> H36M-17 sequence -> VTE+STE`.
  It predicts the centred frame using `t-4…t+4`, repeats boundary frames, and
  currently supports the highest-confidence person only.
- **v4 confidence/occlusion-aware temporal**: the v3 temporal pipeline with
  RTMPose H36M-17 confidence and an all-visible input mask, using the v4
  68-channel `(x,y,confidence,mask)` lifter. It has the same video-only,
  highest-confidence-person boundary as v3.

It uses PySide6 for the interface, OpenCV for media I/O, and the current
PyTorch/MMPose CUDA environment for inference. It does not access a camera.

## Install and run

From the MMPose source root, activate the existing CUDA environment and install
the GUI dependency:

```bash
python -m pip install -r projects/pose_desktop/requirements-desktop.txt
python projects/pose_desktop/main.py --device cuda:0
```

Defaults expect locally untracked weights at `artifacts/models/`, the v2
checkpoint at `work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/`, the v3
checkpoint at `work_dirs/strided_transformer_h36m_rtmpose_9frm/`, and the v4
checkpoint at `work_dirs/strided_transformer_h36m_rtmpose_occconf_9frm/`. Override
weight locations when needed:

```bash
python projects/pose_desktop/main.py --device cuda:0 \
  --det-checkpoint /path/to/rtmdet_m.pth \
  --pose2d-checkpoint /path/to/rtmpose_m.pth \
  --single-lifter-checkpoint /path/to/best_MPJPE_epoch_75.pth \
  --temporal-lifter-checkpoint /path/to/v3_best_MPJPE_epoch_70.pth \
  --occlusion-lifter-checkpoint /path/to/v4_best_MPJPE_epoch_70.pth
```

## Batch input and output layout

Use **Add files** to select images/videos, or **Add folder** to recursively
scan an input directory. Choose an output root; by default it is
`artifacts/batch_output/`. Every run creates a new timestamped directory so
prior results are never overwritten:

```text
artifacts/batch_output/
└── batch_YYYYMMDD_HHMMSS/
    ├── images/       # <index>_<source-name>_pose.<original image extension>
    ├── videos/       # <index>_<source-name>_pose.mp4
    ├── keypoints/    # image JSON and per-video JSONL H36M-17 predictions
    └── manifest.json # run options, source paths, output paths, failures
```

The GUI independently selects whether output media shows the 2D skeleton,
person bounding box, and appended 3D panel. It can also disable keypoint JSON
export. A model selector chooses v2, v3, or v4. If a temporal model is
selected, an image is reported as unsupported rather than silently pretending a
repeated image is a temporal sequence; video JSONL records its noncausal
9-frame mode and window extent. v4 receives RTMPose per-joint confidence and
uses an all-visible mask at deployment; it does not synthetically corrupt user
videos. The 3D panel visualizes the largest detected person in each frame.
It has a white background and a camera-facing H36M coordinate triad: `X` is
image-horizontal, `-Y` is image-up, and `Z` is relative depth. This preserves
the source image's head-up/left-right view while retaining a small depth offset.
Without calibrated camera intrinsics, it is an orientation match rather than a
pixel-exact projection onto the original image.

## Design boundaries

- Model loading and all CUDA calls occur in `BatchInferenceWorker`, never in the
  Qt GUI thread.
- A batch job processes files sequentially, preserving the input list order.
- The 3D panel uses OpenCV drawing rather than Matplotlib, so image and video
  output does not create a heavyweight plotting process for every frame.
- The output is a root-relative pose; it is not calibrated world position or
  metric depth. v3 and v4 are noncausal, so a streaming implementation would
  incur a four-frame latency.

No model weights, recordings, predictions, or logs belong in this directory or
in Git. Keep them under ignored `artifacts/` or `work_dirs/` paths.
