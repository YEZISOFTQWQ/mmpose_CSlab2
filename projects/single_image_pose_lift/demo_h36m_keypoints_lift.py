#!/usr/bin/env python3
"""Visualize a trained H36M keypoints-only pose lifter on one paired sample."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmpose.apis import init_model
from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules


EDGES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
         (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
         (8, 14), (14, 15), (15, 16))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def draw_2d(image, keypoints):
    image = image.copy()
    for start, end in EDGES:
        cv2.line(image, tuple(keypoints[start].astype(int)),
                 tuple(keypoints[end].astype(int)), (0, 255, 0), 3)
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 4, (255, 80, 0), -1)
    return image


def plot_pose(ax, pose, title):
    # H36M poses are in camera coordinates: X is image-horizontal, Y points
    # down in the image and Z is depth. Reorder to X/depth/-Y so the rendered
    # person has the same head-up orientation as the source image.
    pose = pose - pose[0]
    pose = pose[:, [0, 2, 1]]
    pose[:, 2] *= -1
    for start, end in EDGES:
        segment = pose[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], c='#2678b2', lw=2)
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=np.arange(17),
               cmap='viridis', s=24, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / depth (m)')
    ax.set_zlabel('-Y / up (m)')
    # Nearly camera-facing, while retaining enough depth perspective.
    ax.view_init(elev=6, azim=-84)
    radius = max(0.9, float(np.max(np.ptp(pose, axis=0))) * .62)
    center = pose.mean(axis=0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.test_dataloader.dataset.copy()
    # Inference input, 2D overlay and 3D reference must be the same split.
    # ``ann_file`` is made absolute so Human36mDataset does not prepend its
    # configured data_root a second time.
    dataset_cfg['ann_file'] = str(Path(args.ann_file).resolve())
    dataset_cfg['indices'] = [args.index]
    sample = DATASETS.build(dataset_cfg)[0]
    model = init_model(args.config, args.checkpoint, device=args.device)
    prediction = model.test_step(pseudo_collate([sample]))[0]
    pred_3d = prediction.pred_instances.keypoints[0]

    with np.load(args.ann_file) as data:
        keypoints_2d = data['part'][args.index, :, :2]
        gt_3d = data['S'][args.index, :, :3]
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(draw_2d(image, keypoints_2d), cv2.COLOR_BGR2RGB)

    figure = plt.figure(figsize=(17, 6), constrained_layout=True)
    axis = figure.add_subplot(1, 3, 1)
    axis.imshow(image)
    axis.set_title('S1 image + GT 2D keypoints')
    axis.axis('off')
    plot_pose(figure.add_subplot(1, 3, 2, projection='3d'), pred_3d,
              'Predicted 3D (trained lifter)')
    plot_pose(figure.add_subplot(1, 3, 3, projection='3d'), gt_3d,
              'Ground-truth 3D')
    figure.savefig(args.output, dpi=180)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
