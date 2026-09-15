# Joint Angle Visualization (2D)

Multi-person 2D joint-angle analysis on top of RTMDet + RTMPose.
Given an RGB image or video, the scripts detect people, estimate COCO-17
keypoints, compute elbow / knee / hip angles, overlay results on the media,
and (for video) export a time-series curve with CSV / text statistics.

This project does **not** retrain the pose model. It reuses official
pretrained RTMPose weights and the MMPose APIs already present in this
repository.

## Scripts

| Script | Device | Detector / Pose | Overlay style |
|--------|--------|-----------------|---------------|
| `angel_clear.py` | CPU (default) | RTMDet-nano + RTMPose-t | Skeleton, person id `P0/P1…`, per-joint angles; **no** top-left black panel |
| `angel_gpu_clear.py` | CUDA `cuda:0` (default) | RTMDet-m + RTMPose-m | Same clear overlay; GPU defaults aligned with the team README |

Both scripts process **all** detected people (up to `--max-people`).

## Layout in this repository

```text
mmpose_CSlab2/                          # repository / MMPose root
└── projects/mmpose_visualization/
    ├── angel_clear.py
    ├── angel_gpu_clear.py
    ├── requirements.txt
    └── README.md
```

Configs and demos are resolved from the **repository root**
(`demo/mmdetection_cfg/…`, `configs/body_2d_keypoint/rtmpose/…`).
Scripts resolve the repo root as two levels above this folder
(`projects/mmpose_visualization/` → repository root).

## Environment

Use the same Conda / PyTorch / MMPose environment as the rest of the course
project. Then install the small extra list if needed:

```bash
cd /path/to/mmpose_CSlab2
conda activate mmpose
python -m pip install -r projects/mmpose_visualization/requirements.txt
```

- CPU script: any working `torch` + MMPose install.
- GPU script: CUDA-enabled PyTorch; if CUDA is unavailable the script exits
  with an explicit error (use `angel_clear.py` instead).

## Quick start

Run commands from the **repository root**.

### Image

```bash
# CPU
python projects/mmpose_visualization/angel_clear.py \
  --input path/to/image.jpg \
  --output-dir projects/mmpose_visualization/outputs

# GPU
python projects/mmpose_visualization/angel_gpu_clear.py \
  --input path/to/image.jpg \
  --output-dir projects/mmpose_visualization/outputs_gpu
```

### Video

```bash
# CPU (optional --stride 2 for speed)
python projects/mmpose_visualization/angel_clear.py \
  --input path/to/video.mp4 \
  --output-dir projects/mmpose_visualization/outputs \
  --stride 2

# GPU (default stride=1)
python projects/mmpose_visualization/angel_gpu_clear.py \
  --input path/to/video.mp4 \
  --output-dir projects/mmpose_visualization/outputs_gpu
```

First run downloads pretrained checkpoints via HTTP (OpenMMLab URLs).

## Outputs

| File | When | Content |
|------|------|---------|
| `angle_image.jpg` | image mode | Annotated still image |
| `angle_video.mp4` | video mode | Annotated video |
| `angle_curve.png` | video mode only | Joint angle vs. time (tracks `P0…P{N-1}`, `N=--curve-people`) |
| `angle_series.csv` | video mode only | Per-frame angles |
| `angle_stats.txt` | video mode only | min / max / mean / std |

Image mode does **not** write curve / CSV / stats. Re-running into the same
`--output-dir` overwrites previous files with the same names.

## Angle definition

For joint vertex `B` with neighbors `A` and `C`:

```text
theta = arccos( (A-B)·(C-B) / (|A-B| |C-B|) )
```

| Name | A | B (vertex) | C |
|------|---|------------|---|
| L/R Elbow | shoulder | elbow | wrist |
| L/R Knee | hip | knee | ankle |
| L/R Hip | shoulder | hip | knee |

Keypoints below `--kpt-thr` (default `0.3`) are skipped.

## Useful arguments

| Argument | Meaning |
|----------|---------|
| `--device` | `cpu` or `cuda:0` |
| `--stride` | Infer every N-th frame; intermediate frames reuse last pose |
| `--max-frames` | Cap processed frames (`0` = all) |
| `--max-people` | Cap people drawn / analyzed |
| `--curve-people` | How many top-scoring people appear in the curve (default 2) |
| `--bbox-thr` / `--kpt-thr` | Detection / keypoint score thresholds |

## Notes for git

Do **not** commit large images, videos, checkpoints, or `outputs/` folders.
Commit only the Python scripts, `requirements.txt`, and this README.
