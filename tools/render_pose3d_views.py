#!/usr/bin/env python3
"""Render an MMPose 3D prediction from complementary viewing angles."""

import argparse

import cv2
import json_tricks as json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


# COCO-17 skeleton edges, using zero-based keypoint indices.
SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16))


def set_equal_3d_axes(ax, points):
    """Use identical scale in all dimensions so depth is not distorted."""
    center = points.mean(axis=0)
    half_range = max(np.ptp(points, axis=0).max() / 2, 0.1) * 1.15
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def draw_skeleton_3d(ax, points, colors):
    for a, b in SKELETON:
        ax.plot(*points[[a, b]].T, color='#53616b', linewidth=2.2, zorder=1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
               s=54, edgecolors='white', linewidths=0.7, zorder=2)


def draw_projection(ax, points, axes, colors, title, xlabel, ylabel):
    for a, b in SKELETON:
        ax.plot(points[[a, b], axes[0]], points[[a, b], axes[1]],
                color='#53616b', linewidth=2.2, zorder=1)
    ax.scatter(points[:, axes[0]], points[:, axes[1]], c=colors, s=48,
               edgecolors='white', linewidths=0.7, zorder=2)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_json')
    parser.add_argument('image')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(args.prediction_json) as f:
        prediction = json.load(f)[0]
    points = np.asarray(prediction['keypoints'], dtype=float)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    # In MMPose pose-lifting output Y is relative depth and Z is height.
    depth = points[:, 1]
    norm = Normalize(vmin=depth.min(), vmax=depth.max())
    colors = cm.plasma(norm(depth))

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    image_ax = fig.add_subplot(2, 2, 1)
    image_ax.imshow(image)
    image_ax.set_title('Input image', fontsize=12, weight='bold')
    image_ax.axis('off')

    perspective = fig.add_subplot(2, 2, 2, projection='3d')
    draw_skeleton_3d(perspective, points, colors)
    set_equal_3d_axes(perspective, points)
    perspective.view_init(elev=18, azim=-58)
    perspective.set_title('3D perspective', fontsize=12, weight='bold')
    perspective.set_xlabel('X (left / right)')
    perspective.set_ylabel('Y (depth)')
    perspective.set_zlabel('Z (height)')

    front = fig.add_subplot(2, 2, 3)
    draw_projection(front, points, (0, 2), colors, 'Front view', 'X (left / right)',
                    'Z (height)')

    side = fig.add_subplot(2, 2, 4)
    draw_projection(side, points, (1, 2), colors, 'Side view — depth is visible',
                    'Y (near ← → far)', 'Z (height)')

    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='plasma'),
                            ax=[perspective, front, side], shrink=.8, pad=.02)
    colorbar.set_label('Relative depth: purple = nearer, yellow = farther')
    fig.suptitle('MMPose MotionBERT: relative 3D pose', fontsize=16, weight='bold')
    fig.savefig(args.output, dpi=180, bbox_inches='tight')


if __name__ == '__main__':
    main()
