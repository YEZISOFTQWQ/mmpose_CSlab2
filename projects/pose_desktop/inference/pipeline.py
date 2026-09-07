"""RTMDet -> RTMPose -> custom TCN pose-lifter inference pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from config import ModelPaths, RuntimeOptions


@dataclass
class PosePrediction:
    bbox: np.ndarray
    score: float
    keypoints_2d: np.ndarray  # H36M-17, shape (17, 2)
    keypoints_3d: np.ndarray  # root-relative H36M-17, shape (17, 3)


class PosePipeline:
    """Loads models once and performs inference on an OpenCV BGR frame."""

    def __init__(self, paths: ModelPaths, options: RuntimeOptions):
        self.paths = paths
        self.options = options
        self.detector = None
        self.pose2d = None
        self.lifter = None

    def load(self) -> None:
        missing = [str(path) for path in self.paths.__dict__.values()
                   if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError('Missing model/config files:\n' +
                                    '\n'.join(missing))

        # Keep heavy OpenMMLab imports out of GUI startup and worker tests.
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
        self.lifter = init_model(str(self.paths.lifter_config),
                                 str(self.paths.lifter_checkpoint),
                                 device=self.options.device)

    def predict(self, frame_bgr: np.ndarray) -> List[PosePrediction]:
        if self.detector is None:
            raise RuntimeError('PosePipeline.load() must be called first')

        from mmengine.structures import InstanceData
        from mmdet.apis import inference_detector
        from mmpose.apis import (convert_keypoint_definition,
                                 inference_pose_lifter_model, inference_topdown)
        from mmpose.structures import PoseDataSample

        height, width = frame_bgr.shape[:2]
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

        predictions = []
        for track_id, (bbox, det_index, pose_result) in enumerate(
                zip(bboxes, people, pose_results)):
            coco = pose_result.pred_instances.cpu().numpy().keypoints[0]
            h36m = convert_keypoint_definition(coco[None], 'coco', 'h36m')
            sample = PoseDataSample()
            sample.pred_instances = InstanceData(keypoints=h36m,
                                                 bboxes=bbox[None])
            sample.gt_instances = InstanceData()
            sample.track_id = track_id
            lifted = inference_pose_lifter_model(
                self.lifter, [[sample]], with_track_id=True,
                image_size=(width, height),
                norm_pose_2d=self.options.norm_pose_2d)[0]
            pose3d = lifted.pred_instances.keypoints
            while pose3d.ndim > 2:
                pose3d = pose3d[0]
            predictions.append(PosePrediction(
                bbox=bbox.astype(np.float32), score=float(instances.scores[det_index]),
                keypoints_2d=h36m[0].astype(np.float32),
                keypoints_3d=pose3d.astype(np.float32)))
        return predictions
