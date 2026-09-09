"""Input-only synthetic occlusion augmentation for temporal 2D pose lifting."""

from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TemporalJointOcclusion(BaseTransform):
    """Hide joint groups over consecutive frames while retaining 3D labels.

    This transform modifies only detector-style 2D input evidence. It writes a
    binary mask to ``input_occlusion_mask`` and sets the corresponding 2D
    confidence to zero; 3D targets and their visibility weights are untouched.
    The model can therefore learn to reconstruct a corrupted joint from body
    structure and neighbouring time steps rather than treating it as a missing
    supervision label.
    """

    _GROUPS = (
        (1, 2, 3), (4, 5, 6),       # right / left leg
        (11, 12, 13), (14, 15, 16), # right / left arm
        (7, 8, 9, 10),              # torso and head
    )

    def __init__(self,
                 probability: float = .75,
                 span_lengths: Sequence[int] = (1, 2, 4),
                 groups_per_window: int = 1):
        if not 0 <= probability <= 1:
            raise ValueError('probability must lie in [0, 1]')
        if not span_lengths or any(length < 1 for length in span_lengths):
            raise ValueError('span_lengths must contain positive integers')
        if groups_per_window < 1:
            raise ValueError('groups_per_window must be positive')
        self.probability = probability
        self.span_lengths = tuple(int(length) for length in span_lengths)
        self.groups_per_window = groups_per_window

    def transform(self, results: dict) -> dict:
        keypoints = np.asarray(results['keypoints'], dtype=np.float32).copy()
        confidence = np.asarray(
            results['keypoints_visible'], dtype=np.float32).copy()
        if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
            raise ValueError(f'Expected (T, K, 2) keypoints, got {keypoints.shape}')
        if confidence.shape != keypoints.shape[:2]:
            raise ValueError('keypoints_visible must match temporal joint shape')

        frames, joints = keypoints.shape[:2]
        mask = np.ones((frames, joints), dtype=np.float32)
        if np.random.random() >= self.probability:
            results['input_occlusion_mask'] = mask
            return results

        group_count = min(self.groups_per_window, len(self._GROUPS))
        selected = np.random.choice(len(self._GROUPS), size=group_count,
                                    replace=False)
        for group_index in np.atleast_1d(selected):
            span = min(int(np.random.choice(self.span_lengths)), frames)
            start = int(np.random.randint(0, frames - span + 1))
            group = np.asarray(
                [joint for joint in self._GROUPS[group_index] if joint < joints],
                dtype=np.int64)
            if not len(group):
                continue
            frame_indices = np.arange(start, start + span, dtype=np.int64)
            # Coordinates deliberately carry no valid visual evidence. The
            # separate confidence and mask channels tell the network to avoid
            # interpreting the zero-filled value as a true joint location.
            keypoints[np.ix_(frame_indices, group)] = 0
            confidence[np.ix_(frame_indices, group)] = 0
            mask[np.ix_(frame_indices, group)] = 0

        results['keypoints'] = keypoints
        results['keypoints_visible'] = confidence
        results['input_occlusion_mask'] = mask
        return results
