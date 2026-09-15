#!/usr/bin/env python3
"""多人关节角度计算与可视化（GPU / 清爽版）。

基于 angel_gpu_multi.py：对每个人分别算肘/膝/髋角度并叠加骨架与关节文字，
但不绘制左上角黑色汇总面板。

  - 默认 device=cuda:0
  - 默认 RTMDet-m + RTMPose-m（与组长 README 对齐）
  - 视频默认 stride=1 逐帧推理

用法（在仓库根目录 mmpose_CSlab2/ 下执行）：
  conda activate mmpose
  python projects/mmpose_visualization/angel_gpu_clear.py \\
      --input path/to/image.jpg --output-dir projects/mmpose_visualization/outputs
  python projects/mmpose_visualization/angel_gpu_clear.py \\
      --input path/to/video.mp4 --output-dir projects/mmpose_visualization/outputs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmengine.logging import print_log

# projects/mmpose_visualization/*.py -> repository root (MMPose source tree)
PROJECT_DIR = Path(__file__).resolve().parent
MMPOSE_ROOT = PROJECT_DIR.parent.parent
ROOT = PROJECT_DIR
if not (MMPOSE_ROOT / "mmpose").is_dir() or not (MMPOSE_ROOT / "configs").is_dir():
    raise RuntimeError(
        f"Cannot locate MMPose repository root from {PROJECT_DIR}. "
        f"Expected configs/ and mmpose/ under {MMPOSE_ROOT}."
    )
if str(MMPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(MMPOSE_ROOT))

from mmpose.apis import inference_topdown  # noqa: E402
from mmpose.apis import init_model as init_pose_estimator  # noqa: E402
from mmpose.evaluation.functional import nms  # noqa: E402
from mmpose.structures import merge_data_samples  # noqa: E402
from mmpose.utils import adapt_mmdet_pipeline  # noqa: E402

from mmdet.apis import inference_detector, init_detector  # noqa: E402

KPT = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

ANGLE_DEFS: List[Tuple[str, str, str, str]] = [
    ("L_Elbow", "l_shoulder", "l_elbow", "l_wrist"),
    ("R_Elbow", "r_shoulder", "r_elbow", "r_wrist"),
    ("L_Knee", "l_hip", "l_knee", "l_ankle"),
    ("R_Knee", "r_hip", "r_knee", "r_ankle"),
    ("L_Hip", "l_shoulder", "l_hip", "l_knee"),
    ("R_Hip", "r_shoulder", "r_hip", "r_knee"),
]

ANGLE_COLORS = {
    "L_Elbow": (0, 200, 255),
    "R_Elbow": (0, 140, 255),
    "L_Knee": (80, 220, 100),
    "R_Knee": (40, 180, 60),
    "L_Hip": (255, 180, 80),
    "R_Hip": (255, 120, 40),
}

PERSON_COLORS = [
    (200, 200, 200),
    (255, 180, 100),
    (180, 120, 255),
    (100, 220, 220),
    (120, 180, 255),
    (200, 255, 120),
]


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "请求使用 GPU（{}），但当前 PyTorch 检测不到 CUDA。\n"
                "请改用 CPU 清爽版脚本：python angel_clear.py ...\n"
                "当前 torch={}, cuda={}".format(
                    torch.__version__, torch.version.cuda
                )
            )
        return "cuda:0" if requested == "cuda" else requested
    return requested


def angle_at_vertex(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Optional[float]:
    ba = a.astype(np.float64) - b.astype(np.float64)
    bc = c.astype(np.float64) - b.astype(np.float64)
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cos_v = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def compute_angles(
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.3,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for name, p1, p2, p3 in ANGLE_DEFS:
        i1, i2, i3 = KPT[p1], KPT[p2], KPT[p3]
        if min(scores[i1], scores[i2], scores[i3]) < kpt_thr:
            out[name] = None
            continue
        out[name] = angle_at_vertex(keypoints[i1], keypoints[i2], keypoints[i3])
    return out


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    color: Tuple[int, int, int],
    kpt_thr: float = 0.3,
) -> None:
    links = [
        ("l_shoulder", "r_shoulder"),
        ("l_shoulder", "l_elbow"),
        ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"),
        ("r_elbow", "r_wrist"),
        ("l_shoulder", "l_hip"),
        ("r_shoulder", "r_hip"),
        ("l_hip", "r_hip"),
        ("l_hip", "l_knee"),
        ("l_knee", "l_ankle"),
        ("r_hip", "r_knee"),
        ("r_knee", "r_ankle"),
    ]
    for a, b in links:
        ia, ib = KPT[a], KPT[b]
        if scores[ia] < kpt_thr or scores[ib] < kpt_thr:
            continue
        pa = tuple(map(int, keypoints[ia]))
        pb = tuple(map(int, keypoints[ib]))
        cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    for name, idx in KPT.items():
        if name == "nose":
            continue
        if scores[idx] < kpt_thr:
            continue
        pt = tuple(map(int, keypoints[idx]))
        cv2.circle(img, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_person_angles(
    img: np.ndarray,
    person_id: int,
    keypoints: np.ndarray,
    scores: np.ndarray,
    angles: Dict[str, Optional[float]],
    kpt_thr: float = 0.3,
) -> None:
    color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
    draw_skeleton(img, keypoints, scores, color, kpt_thr)

    if scores[KPT["nose"]] >= kpt_thr:
        anchor = keypoints[KPT["nose"]]
    elif scores[KPT["l_shoulder"]] >= kpt_thr:
        anchor = keypoints[KPT["l_shoulder"]]
    else:
        anchor = keypoints[KPT["r_shoulder"]]
    ax, ay = int(anchor[0]), int(anchor[1])
    cv2.putText(
        img,
        f"P{person_id}",
        (ax - 10, max(20, ay - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    for name, p1, p2, p3 in ANGLE_DEFS:
        val = angles.get(name)
        if val is None:
            continue
        idx = KPT[p2]
        if scores[idx] < kpt_thr:
            continue
        x, y = map(int, keypoints[idx])
        acolor = ANGLE_COLORS.get(name, (255, 255, 255))
        cv2.putText(
            img,
            f"{name}:{val:.1f}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            acolor,
            1,
            cv2.LINE_AA,
        )


def draw_all_people(
    img_bgr: np.ndarray,
    people_data: List[Tuple[int, np.ndarray, np.ndarray, Dict[str, Optional[float]]]],
    kpt_thr: float = 0.3,
) -> np.ndarray:
    """只画骨架 / 编号 / 关节角，不画左上角黑框面板。"""
    vis = img_bgr.copy()
    for pid, kpts, scores, angles in people_data:
        draw_person_angles(vis, pid, kpts, scores, angles, kpt_thr)
    return vis


def detect_and_pose_multi(
    img,
    detector,
    pose_estimator,
    det_cat_id: int = 0,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    kpt_thr: float = 0.3,
    max_people: int = 10,
) -> List[Tuple[int, np.ndarray, np.ndarray, Dict[str, Optional[float]]]]:
    det_result = inference_detector(detector, img)
    pred = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)
    mask = np.logical_and(pred.labels == det_cat_id, pred.scores > bbox_thr)
    bboxes = bboxes[mask]
    if len(bboxes) == 0:
        return []
    bboxes = bboxes[nms(bboxes, nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)
    instances = data_samples.get("pred_instances", None)
    if instances is None or len(instances) == 0:
        return []

    kpts_all = instances.keypoints
    scores_all = instances.keypoint_scores
    mean_scores = scores_all.mean(axis=1)
    order = np.argsort(-mean_scores)

    people = []
    for rank, idx in enumerate(order):
        if rank >= max_people:
            break
        kpts = kpts_all[idx]
        scores = scores_all[idx]
        angles = compute_angles(kpts, scores, kpt_thr)
        people.append((rank, kpts, scores, angles))
    return people


def summarize_series(series: Dict[str, List[float]]) -> Dict[str, dict]:
    stats = {}
    for name, vals in series.items():
        if not vals:
            stats[name] = {}
            continue
        arr = np.array(vals, dtype=np.float64)
        stats[name] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "count": int(len(arr)),
        }
    return stats


def plot_angle_curves(
    frame_ids: List[int],
    series: Dict[str, List[Optional[float]]],
    out_path: Path,
    fps: float = 30.0,
    title: str = "Joint Angle over Time (multi GPU)",
) -> None:
    plt.figure(figsize=(11, 5.5))
    t = np.array(frame_ids, dtype=np.float64) / max(fps, 1e-6)
    plotted = False
    for key, vals in series.items():
        if not vals:
            continue
        y = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
        if np.all(np.isnan(y)):
            continue
        joint = key.split("_", 1)[-1] if "_" in key else key
        if joint in ANGLE_COLORS:
            b, g, r = ANGLE_COLORS[joint]
            color = (r / 255, g / 255, b / 255)
        else:
            color = None
        plt.plot(t, y, label=key, color=color, linewidth=1.6)
        plotted = True

    if not plotted:
        plt.text(0.5, 0.5, "No angle data", ha="center", transform=plt.gca().transAxes)

    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print_log(f"Saved angle curve: {out_path}", logger="current")


def save_csv(
    frame_ids: List[int],
    series: Dict[str, List[Optional[float]]],
    out_path: Path,
) -> None:
    keys = sorted(series.keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + keys)
        for i, fid in enumerate(frame_ids):
            row = [fid]
            for key in keys:
                v = series[key][i] if i < len(series[key]) else None
                row.append("" if v is None else f"{v:.3f}")
            writer.writerow(row)


def build_models(args):
    print_log(f"Loading detector on {args.device} ...", logger="current")
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    print_log(f"Loading pose estimator on {args.device} ...", logger="current")
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )
    if args.device.startswith("cuda"):
        print_log(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
            logger="current",
        )
    return detector, pose_estimator


def process_image(args, detector, pose_estimator) -> None:
    img = mmcv.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    people = detect_and_pose_multi(
        args.input,
        detector,
        pose_estimator,
        args.det_cat_id,
        args.bbox_thr,
        args.nms_thr,
        args.kpt_thr,
        args.max_people,
    )
    if not people:
        raise RuntimeError("No person detected in the image.")

    vis = draw_all_people(img, people, args.kpt_thr)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "angle_image.jpg"
    cv2.imwrite(str(out_img), vis)
    print_log(f"Detected {len(people)} person(s). Saved: {out_img}", logger="current")
    for pid, _, _, angles in people:
        print_log(f"--- P{pid} ---", logger="current")
        for k, v in angles.items():
            print_log(
                f"  {k}: {v if v is None else f'{v:.2f} deg'}", logger="current"
            )


def process_video(args, detector, pose_estimator) -> None:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "angle_video.mp4"
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    joint_names = [d[0] for d in ANGLE_DEFS]
    curve_pids = list(range(max(1, args.curve_people)))
    series: Dict[str, List[Optional[float]]] = defaultdict(list)
    valid_series: Dict[str, List[float]] = defaultdict(list)
    frame_ids: List[int] = []

    frame_idx = 0
    inferred = 0
    last_people: List[
        Tuple[int, np.ndarray, np.ndarray, Dict[str, Optional[float]]]
    ] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        need_infer = (frame_idx % max(args.stride, 1) == 0) or (not last_people)
        if need_infer:
            last_people = detect_and_pose_multi(
                frame,
                detector,
                pose_estimator,
                args.det_cat_id,
                args.bbox_thr,
                args.nms_thr,
                args.kpt_thr,
                args.max_people,
            )
            inferred += 1

        vis = (
            draw_all_people(frame, last_people, args.kpt_thr)
            if last_people
            else frame
        )
        writer.write(vis)
        frame_ids.append(frame_idx)

        angles_by_pid = {pid: ang for pid, _, _, ang in last_people}
        for pid in curve_pids:
            ang = angles_by_pid.get(pid, {})
            for jn in joint_names:
                key = f"P{pid}_{jn}"
                v = ang.get(jn) if ang else None
                series[key].append(v)
                if v is not None and need_infer:
                    valid_series[key].append(v)

        if inferred % 20 == 0 or (need_infer and inferred == 1):
            print_log(
                f"Video progress: frame {frame_idx}/{total} "
                f"(inferred {inferred}, people={len(last_people)})",
                logger="current",
            )
        frame_idx += 1

    cap.release()
    writer.release()
    print_log(f"Saved angled video: {out_video}", logger="current")

    plot_angle_curves(
        frame_ids,
        series,
        out_dir / "angle_curve.png",
        fps=fps,
        title="Joint Angle over Time (multi-person GPU)",
    )
    save_csv(frame_ids, series, out_dir / "angle_series.csv")

    stats = summarize_series(valid_series)
    stats_path = out_dir / "angle_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Joint angle statistics (multi-person GPU, clear/no panel)\n")
        f.write("=" * 40 + "\n")
        for name, s in stats.items():
            f.write(f"{name}: {s}\n")
            print_log(f"{name}: {s}", logger="current")
    print_log(f"Saved stats: {stats_path}", logger="current")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-person RTMPose joint angle (GPU, no panel)"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "angle_results_gpu_clear"),
    )
    parser.add_argument(
        "--det-config",
        type=str,
        default=str(
            MMPOSE_ROOT / "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
        ),
    )
    parser.add_argument(
        "--det-checkpoint",
        type=str,
        default=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
            "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        ),
    )
    parser.add_argument(
        "--pose-config",
        type=str,
        default=str(
            MMPOSE_ROOT
            / "configs/body_2d_keypoint/rtmpose/body8/"
            "rtmpose-m_8xb256-420e_body8-256x192.py"
        ),
    )
    parser.add_argument(
        "--pose-checkpoint",
        type=str,
        default=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--max-people", type=int, default=10)
    parser.add_argument("--curve-people", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "image", "video"],
        default="auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    os.chdir(MMPOSE_ROOT)

    suffix = Path(args.input).suffix.lower()
    mode = args.mode
    if mode == "auto":
        mode = (
            "video"
            if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            else "image"
        )

    input_path = Path(args.input)
    if not input_path.is_absolute():
        cand = (ROOT / input_path).resolve()
        args.input = str(
            cand if cand.exists() else (MMPOSE_ROOT / input_path).resolve()
        )

    out = Path(args.output_dir)
    if not out.is_absolute():
        args.output_dir = str((ROOT / out).resolve())

    print_log(f"Device: {args.device}", logger="current")
    print_log(f"Input: {args.input}", logger="current")
    print_log(f"Output: {args.output_dir}", logger="current")
    print_log(
        f"Mode: multi-person clear/no panel GPU | Det=RTMDet-m | Pose=RTMPose-m | "
        f"torch={torch.__version__}",
        logger="current",
    )

    detector, pose_estimator = build_models(args)
    if mode == "image":
        process_image(args, detector, pose_estimator)
    else:
        process_video(args, detector, pose_estimator)


if __name__ == "__main__":
    main()
