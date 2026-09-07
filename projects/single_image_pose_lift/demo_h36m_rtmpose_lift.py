#!/usr/bin/env python3
"""Run RTMDet + RTMPose 2D inference and the trained 3D lifter on one H36M frame."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmengine.structures import InstanceData
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (convert_keypoint_definition, inference_pose_lifter_model,
                         inference_topdown, init_model)
from mmpose.structures import PoseDataSample
from mmpose.utils import register_all_modules
from demo_h36m_keypoints_lift import draw_2d, plot_pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--pose2d-config', required=True)
    parser.add_argument('--pose2d-checkpoint', required=True)
    parser.add_argument('--pose3d-config', required=True)
    parser.add_argument('--pose3d-checkpoint', required=True)
    parser.add_argument('--ann-file', help='Optional paired H36M annotation NPZ')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', type=int, help='Row index in --ann-file')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.ann_file is None) != (args.index is None):
        raise ValueError('--ann-file and --index must be supplied together')
    register_all_modules()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    height, width = image.shape[:2]

    detector = init_detector(args.det_config, args.det_checkpoint,
                             device=args.device)
    detected = inference_detector(detector, args.image).pred_instances.cpu().numpy()
    person_indices = np.flatnonzero(detected.labels == 0)
    if not len(person_indices):
        raise RuntimeError('No COCO person detected')
    person = person_indices[np.argmax(detected.scores[person_indices])]
    bbox = detected.bboxes[person:person + 1]

    pose2d = init_model(args.pose2d_config, args.pose2d_checkpoint, args.device)
    pose_result = inference_topdown(pose2d, args.image, bbox)[0]
    pred2d = pose_result.pred_instances.cpu().numpy()
    coco = pred2d.keypoints[0]
    h36m = convert_keypoint_definition(coco[None], 'coco', 'h36m')

    # Adapt the single detected person to MMPose's pose-lifter inference API.
    pose_sample = PoseDataSample()
    pose_sample.pred_instances = InstanceData(keypoints=h36m, bboxes=bbox)
    pose_sample.gt_instances = InstanceData()
    pose_sample.track_id = 0
    lifter = init_model(args.pose3d_config, args.pose3d_checkpoint, args.device)
    lift_result = inference_pose_lifter_model(
        lifter, [[pose_sample]], with_track_id=True, image_size=(width, height),
        norm_pose_2d=False)[0]
    pred3d = lift_result.pred_instances.keypoints[0]

    image = cv2.cvtColor(draw_2d(image, h36m[0]), cv2.COLOR_BGR2RGB)
    has_gt = args.ann_file is not None
    figure = plt.figure(figsize=((17 if has_gt else 11), 6), constrained_layout=True)
    axis = figure.add_subplot(1, 3, 1)
    axis.imshow(image)
    axis.set_title('RTMDet + RTMPose-M 2D input')
    axis.axis('off')
    if has_gt:
        with np.load(args.ann_file) as annotation:
            gt3d = annotation['S'][args.index, :, :3]
        error_mm = np.linalg.norm((pred3d - pred3d[0]) - (gt3d - gt3d[0]),
                                  axis=1).mean() * 1000
        plot_pose(figure.add_subplot(1, 3, 2, projection='3d'), pred3d,
                  f'3D from predicted 2D | MPJPE {error_mm:.1f} mm')
        plot_pose(figure.add_subplot(1, 3, 3, projection='3d'), gt3d,
                  'Ground-truth 3D')
    else:
        axis.remove()
        axis = figure.add_subplot(1, 2, 1)
        axis.imshow(image)
        axis.set_title('RTMDet + RTMPose-M 2D input')
        axis.axis('off')
        plot_pose(figure.add_subplot(1, 2, 2, projection='3d'), pred3d,
                  'Predicted 3D from RTMPose 2D')
    figure.savefig(args.output, dpi=180)
    print(f'2D bbox: {bbox[0].round(1).tolist()}')
    if has_gt:
        print(f'Root-relative MPJPE: {error_mm:.2f} mm')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
