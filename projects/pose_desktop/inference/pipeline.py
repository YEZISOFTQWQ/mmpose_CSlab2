"""RTMDet -> RTMPose -> single-frame or temporal 3D lifting pipeline."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Sequence

import numpy as np

from config import ModelPaths, RuntimeOptions, SOURCE_ROOT


@dataclass
class PoseDetection:
    """One H36M-17 2D detection before it is lifted to 3D."""

    bbox: np.ndarray
    score: float
    keypoints_2d: np.ndarray
    keypoint_scores: np.ndarray


@dataclass
class PosePrediction:
    bbox: np.ndarray
    score: float
    keypoints_2d: np.ndarray  # H36M-17, shape (17, 2)
    keypoints_3d: np.ndarray  # root-relative H36M-17, shape (17, 3)


class PosePipeline:
    """Load models once and expose v2, v3 and v4 inference methods."""

    def __init__(self, paths: ModelPaths, options: RuntimeOptions):
        self.paths = paths
        self.options = options
        self.detector = None
        self.pose2d = None
        self.lifter = None

    def load(self) -> None:
        lifter_config, lifter_checkpoint = self.paths.lifter_paths(
            self.options.lifter_mode)
        required = (self.paths.det_config, self.paths.det_checkpoint,
                    self.paths.pose2d_config, self.paths.pose2d_checkpoint,
                    lifter_config, lifter_checkpoint)
        missing = [str(path) for path in required if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError('Missing model/config files:\n' +
                                    '\n'.join(missing))

        # The v3/v4 configs import ``projects.strided_transformer_pose_lift``.
        # Running main.py otherwise puts only this app directory on sys.path.
        if str(SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(SOURCE_ROOT))

        from mmdet.apis import init_detector
        from mmpose.apis import init_model
        from mmpose.utils import adapt_mmdet_pipeline, register_all_modules

        register_all_modules()
        self.detector = init_detector(str(self.paths.det_config),
                                      str(self.paths.det_checkpoint),
                                      device=self.options.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose2d = init_model(str(self.paths.pose2d_config),
                                 str(self.paths.pose2d_checkpoint),
                                 device=self.options.device)
        self.lifter = init_model(str(lifter_config), str(lifter_checkpoint),
                                 device=self.options.device)

    def detect_2d(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        """Detect people and convert RTMPose COCO joints to H36M-17."""
        if self.detector is None:
            raise RuntimeError('PosePipeline.load() must be called first')

        from mmdet.apis import inference_detector
        from mmpose.apis import convert_keypoint_definition, inference_topdown

        detected = inference_detector(self.detector, frame_bgr)
        instances = detected.pred_instances.cpu().numpy()
        people = np.flatnonzero((instances.labels == 0) &
                                (instances.scores >= self.options.bbox_threshold))
        if not len(people):
            return []
        people = people[np.argsort(instances.scores[people])[::-1]]
        people = people[:self.options.max_people]
        bboxes = instances.bboxes[people]
        pose_results = inference_topdown(self.pose2d, frame_bgr, bboxes)

        detections = []
        for bbox, det_index, pose_result in zip(bboxes, people, pose_results):
            pose_instances = pose_result.pred_instances.cpu().numpy()
            coco = pose_instances.keypoints[0]
            coco_scores = pose_instances.keypoint_scores[0]
            # Conversion averages derived joints (pelvis, thorax, spine, head).
            # Supplying score as a third coordinate applies the same mapping to
            # its confidence, which matches the v4 H36M-17 training input.
            coco_with_scores = np.concatenate(
                [coco, coco_scores[:, None]], axis=-1)
            h36m_with_scores = convert_keypoint_definition(
                coco_with_scores[None], 'coco', 'h36m')[0]
            detections.append(PoseDetection(
                bbox=bbox.astype(np.float32),
                score=float(instances.scores[det_index]),
                keypoints_2d=h36m_with_scores[:, :2].astype(np.float32),
                keypoint_scores=np.clip(h36m_with_scores[:, 2], 0., 1.)
                .astype(np.float32)))
        return detections

    @staticmethod
    def _sample_from_detection(detection: PoseDetection, track_id: int):
        """Build the MMPose input object used by the standard v2 lifter."""
        from mmengine.structures import InstanceData
        from mmpose.structures import PoseDataSample

        sample = PoseDataSample()
        sample.pred_instances = InstanceData(
            keypoints=detection.keypoints_2d[None],
            bboxes=detection.bbox[None])
        sample.gt_instances = InstanceData()
        sample.track_id = track_id
        return sample

    def predict(self, frame_bgr: np.ndarray) -> List[PosePrediction]:
        """Run v2 single-frame lifting; valid for images and videos."""
        if self.options.temporal:
            raise RuntimeError('Use predict_temporal() for a temporal model')
        from mmpose.apis import inference_pose_lifter_model

        height, width = frame_bgr.shape[:2]
        predictions = []
        for track_id, detection in enumerate(self.detect_2d(frame_bgr)):
            sample = self._sample_from_detection(detection, track_id)
            lifted = inference_pose_lifter_model(
                self.lifter, [[sample]], with_track_id=True,
                image_size=(width, height),
                norm_pose_2d=self.options.norm_pose_2d)[0]
            pose3d = lifted.pred_instances.keypoints
            while pose3d.ndim > 2:
                pose3d = pose3d[0]
            predictions.append(PosePrediction(
                bbox=detection.bbox, score=detection.score,
                keypoints_2d=detection.keypoints_2d,
                keypoints_3d=pose3d.astype(np.float32)))
        return predictions

    def predict_temporal(self, window: Sequence[Sequence[PoseDetection]],
                         image_size: tuple[int, int]) -> List[PosePrediction]:
        """Lift one centred 9-frame window with v3 or v4.

        v3 was trained on one H36M person per sequence and does not include a
        person-tracking loss. Temporal GUI inference explicitly selects the
        highest-score detection from every frame rather than mixing identities.
        """
        if not self.options.temporal:
            raise RuntimeError('Temporal prediction requires a v3 or v4 model')
        if len(window) != 9:
            raise ValueError(f'v3 requires a 9-frame window, got {len(window)}')
        if any(not detections for detections in window):
            return []

        import torch

        center_detection = window[4][0]
        keypoints = np.stack([detections[0].keypoints_2d for detections in window])
        width, height = image_size
        center = np.asarray((width * .5, height * .5), dtype=np.float32)
        scale = float(width * .5)
        if scale <= 0:
            raise ValueError(f'Invalid frame width: {width}')
        normalized = (keypoints.astype(np.float32) - center) / scale
        if self.options.occlusion_aware:
            confidence = np.stack(
                [detections[0].keypoint_scores for detections in window])
            mask = np.ones(confidence.shape, dtype=np.float32)
            features = np.concatenate(
                [normalized, confidence[..., None], mask[..., None]], axis=-1)
        else:
            features = normalized
        # Match TemporalImagePoseLifting.encode(): (T, J, C) -> (J*C, T).
        inputs = features.transpose(1, 2, 0).reshape(1, -1, 9)
        tensor = torch.from_numpy(inputs).to(self.options.device)
        with torch.inference_mode():
            pose3d = self.lifter(tensor, None, mode='tensor')
        pose3d = pose3d.detach().cpu().numpy()[0]
        # v3 regresses 16 non-root joints. The deployment view is root-relative.
        pose3d = np.insert(pose3d, 0, np.zeros(3, dtype=np.float32), axis=0)
        return [PosePrediction(
            bbox=center_detection.bbox, score=center_detection.score,
            keypoints_2d=center_detection.keypoints_2d,
            keypoints_3d=pose3d.astype(np.float32))]
