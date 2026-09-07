#!/usr/bin/env python3
"""Benchmark RTMPose-to-custom-lifter inference on representative S1 frames.

The H36M annotation supplies the paired camera-space 3D target.  RGB frames
are read from S1.tar only for the selected samples, then the deploy-time path
is used: RTMDet -> RTMPose -> COCO-to-H36M conversion -> trained lifter.
"""

import argparse
import csv
import re
import tarfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mmengine.structures import InstanceData
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (convert_keypoint_definition, inference_pose_lifter_model,
                         inference_topdown, init_model)
from mmpose.structures import PoseDataSample
from mmpose.utils import adapt_mmdet_pipeline, register_all_modules
from demo_h36m_keypoints_lift import draw_2d, plot_pose


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', required=True, type=Path)
    parser.add_argument('--ann-file', required=True, type=Path)
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--pose2d-config', required=True)
    parser.add_argument('--pose2d-checkpoint', required=True)
    parser.add_argument('--pose3d-config', required=True)
    parser.add_argument('--pose3d-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--per-sequence', type=int, default=2,
                        help='Evenly spaced frames for each S1 action/trial.')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--indices', type=int, nargs='+',
                        help='Exact NPZ rows to evaluate; overrides sampling.')
    parser.add_argument('--subjects', nargs='+', default=['S1'])
    parser.add_argument('--norm-pose-2d', action='store_true',
                        help='Use the official pose-lifter bbox normalization.')
    parser.add_argument('--model-label', default='Custom TCN prediction')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def choose_samples(names, per_sequence, subjects):
    """Pick frames across every S1 action/trial from one fixed camera."""
    matcher = re.compile(r'^(' + '|'.join(map(re.escape, subjects))
                         + r')_(.+)\.54138969_\d+\.jpg$')
    groups = {}
    for index, name in enumerate(names):
        match = matcher.match(name)
        if match:
            groups.setdefault((match.group(1), match.group(2)), []).append(index)
    chosen = []
    for sequence, indices in sorted(groups.items()):
        positions = np.linspace(0, len(indices) - 1, per_sequence,
                                dtype=int)
        chosen.extend(indices[p] for p in np.unique(positions))
    return chosen


def extract_images(archive_dir, names, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    by_subject = {}
    for name in names:
        by_subject.setdefault(name.split('_', 1)[0], []).append(name)
    for subject, subject_names in by_subject.items():
        with tarfile.open(archive_dir / f'{subject}.tar') as archive:
            for name in subject_names:
                destination = image_dir / name
                if not destination.exists():
                    with archive.extractfile(name) as source:
                        destination.write_bytes(source.read())


def lift_one(image_path, detector, pose2d, lifter, norm_pose_2d):
    image = cv2.imread(str(image_path))
    detected = inference_detector(detector, str(image_path)).pred_instances.cpu().numpy()
    person_indices = np.flatnonzero((detected.labels == 0)
                                    & (detected.scores >= 0.3))
    if not len(person_indices):
        return image, None, None
    person = person_indices[np.argmax(detected.scores[person_indices])]
    bbox = detected.bboxes[person:person + 1]
    result = inference_topdown(pose2d, str(image_path), bbox)[0]
    coco = result.pred_instances.cpu().numpy().keypoints[0]
    h36m = convert_keypoint_definition(coco[None], 'coco', 'h36m')
    sample = PoseDataSample()
    sample.pred_instances = InstanceData(keypoints=h36m, bboxes=bbox)
    sample.gt_instances = InstanceData()
    sample.track_id = 0
    height, width = image.shape[:2]
    pred = inference_pose_lifter_model(
        lifter, [[sample]], with_track_id=True, image_size=(width, height),
        norm_pose_2d=norm_pose_2d)[0].pred_instances.keypoints
    # TCN returns (1, 17, 3), while MotionBERT retains a singleton temporal
    # dimension (1, 1, 17, 3).  Both represent one person in one target frame.
    while pred.ndim > 2:
        pred = pred[0]
    return image, h36m[0], pred


def main():
    args = parse_args()
    register_all_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / 'images'
    with np.load(args.ann_file) as ann:
        all_names = ann['imgname'].astype(str)
        selected = (args.indices if args.indices is not None else
                    choose_samples(all_names, args.per_sequence, args.subjects))
        names = all_names[selected]
        gt_poses = ann['S'][selected, :, :3]
    extract_images(args.archive_dir, names, image_dir)

    detector = init_detector(args.det_config, args.det_checkpoint, args.device)
    # This follows MMPose's official 3D demo.  The active MMPose registry
    # otherwise makes MMDetection's PackDetInputs unavailable at inference.
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose2d = init_model(args.pose2d_config, args.pose2d_checkpoint, args.device)
    lifter = init_model(args.pose3d_config, args.pose3d_checkpoint, args.device)
    records = []
    for order, (index, name, gt) in enumerate(zip(selected, names, gt_poses), 1):
        image, keypoints, pred = lift_one(image_dir / name, detector, pose2d,
                                           lifter, args.norm_pose_2d)
        if pred is None:
            print(f'[{order}/{len(names)}] no person: {name}')
            continue
        error_mm = np.linalg.norm((pred - pred[0]) - (gt - gt[0]), axis=1).mean() * 1000
        records.append(dict(index=int(index), name=name, mpjpe_mm=float(error_mm),
                            image=image, keypoints=keypoints, pred=pred, gt=gt))
        print(f'[{order}/{len(names)}] {name}: {error_mm:.1f} mm')

    records.sort(key=lambda item: item['mpjpe_mm'])
    with (args.output_dir / 'ranking.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['rank', 'index', 'name', 'mpjpe_mm'])
        writer.writeheader()
        for rank, item in enumerate(records, 1):
            writer.writerow(dict(rank=rank, index=item['index'], name=item['name'],
                                 mpjpe_mm=f"{item['mpjpe_mm']:.3f}"))

    chosen = records[:args.top_k]
    figure = plt.figure(figsize=(16, 5 * len(chosen)), constrained_layout=True)
    for row, item in enumerate(chosen):
        image = cv2.cvtColor(draw_2d(item['image'], item['keypoints']),
                             cv2.COLOR_BGR2RGB)
        axis = figure.add_subplot(len(chosen), 3, row * 3 + 1)
        axis.imshow(image)
        axis.set_title(f"#{row + 1} {item['name']}\nRTMPose 2D input")
        axis.axis('off')
        plot_pose(figure.add_subplot(len(chosen), 3, row * 3 + 2,
                                     projection='3d'), item['pred'],
                  f"{args.model_label} | {item['mpjpe_mm']:.1f} mm")
        plot_pose(figure.add_subplot(len(chosen), 3, row * 3 + 3,
                                     projection='3d'), item['gt'],
                  'Human3.6M ground truth')
    output = args.output_dir / 'top_predictions_vs_gt.png'
    figure.savefig(output, dpi=150)
    print(f'completed {len(records)}/{len(names)} frames')
    print(f'saved {output}')
    print(f'saved {args.output_dir / "ranking.csv"}')


if __name__ == '__main__':
    main()
