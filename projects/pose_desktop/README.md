# MMPose batch inference desktop application

This desktop GUI runs the second project model over selected images and videos:

`input image/video -> RTMDet-M -> RTMPose-M -> H36M-17 conversion -> custom TCN -> annotated output`.

It uses PySide6 for the interface, OpenCV for media I/O, and the current
PyTorch/MMPose CUDA environment for inference. It does not access a camera.

## Install and run

From the MMPose source root, activate the existing CUDA environment and install
the GUI dependency:

```bash
python -m pip install -r projects/pose_desktop/requirements-desktop.txt
python projects/pose_desktop/main.py --device cuda:0
```

Defaults expect locally untracked weights at `artifacts/models/` and the second
TCN checkpoint at `work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/`. Override
weight locations when needed:

```bash
python projects/pose_desktop/main.py --device cuda:0 \
  --det-checkpoint /path/to/rtmdet_m.pth \
  --pose2d-checkpoint /path/to/rtmpose_m.pth \
  --lifter-checkpoint /path/to/best_MPJPE_epoch_75.pth
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
export. The 3D panel visualizes the largest detected person in each frame.
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
- The output is a root-relative, single-frame pose; it is not calibrated world
  position or metric depth.

No model weights, recordings, predictions, or logs belong in this directory or
in Git. Keep them under ignored `artifacts/` or `work_dirs/` paths.
