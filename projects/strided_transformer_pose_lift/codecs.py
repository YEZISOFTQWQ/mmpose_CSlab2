"""Keypoint codec for fixed-window, detector-matched pose lifting."""

from typing import List, Optional, Tuple, Union

import numpy as np

from mmpose.codecs.base import BaseKeypointCodec
from mmpose.registry import KEYPOINT_CODECS


@KEYPOINT_CODECS.register_module()
class TemporalImagePoseLifting(BaseKeypointCodec):
    """Encode a 2D window and both supervision scales for temporal lifting.

    The input is normalized with the source image dimensions, rather than
    camera intrinsics. It therefore works with the existing RTMDet + RTMPose
    detections and does not require H36M's unavailable ``cameras.pkl``.
    Targets are root-relative H36M coordinates in metres.
    """

    auxiliary_encode_keys = {
        'lifting_target', 'lifting_target_visible', 'keypoints_3d',
        'keypoints_3d_visible', 'input_occlusion_mask'
    }
    instance_mapping_table = dict(
        lifting_target='lifting_target',
        lifting_target_visible='lifting_target_visible')
    label_mapping_table = dict(
        trajectory_weights='trajectory_weights',
        lifting_target_label='lifting_target_label',
        lifting_target_weight='lifting_target_weight',
        lifting_sequence_label='lifting_sequence_label',
        lifting_sequence_weight='lifting_sequence_weight')

    def __init__(self,
                 num_keypoints: int,
                 image_size: Tuple[int, int] = (1000, 1002),
                 root_index: Union[int, List[int]] = 0,
                 remove_root: bool = True,
                 save_index: bool = True,
                 concat_vis: bool = False,
                 concat_mask: bool = False):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.image_size = np.asarray(image_size, dtype=np.float32)
        self.root_index = [root_index] if isinstance(root_index, int) else root_index
        self.remove_root = remove_root
        self.save_index = save_index
        self.concat_vis = concat_vis
        self.concat_mask = concat_mask

    def _root_relative(self, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        root = np.mean(target[..., self.root_index, :], axis=-2,
                       dtype=np.float32)
        target = target - root[..., None, :]
        if self.remove_root:
            if len(self.root_index) != 1:
                raise ValueError('remove_root requires one root index')
            target = np.delete(target, self.root_index[0], axis=-2)
        return target, root

    def _weights(self, visible: Optional[np.ndarray], target: np.ndarray) -> np.ndarray:
        if visible is None:
            visible = np.ones(target.shape[:-1], dtype=np.float32)
        weight = (np.asarray(visible) > .5).astype(np.float32)
        if self.remove_root:
            weight = np.delete(weight, self.root_index[0], axis=-1)
        return weight

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               lifting_target: Optional[np.ndarray] = None,
               lifting_target_visible: Optional[np.ndarray] = None,
               keypoints_3d: Optional[np.ndarray] = None,
               keypoints_3d_visible: Optional[np.ndarray] = None,
               input_occlusion_mask: Optional[np.ndarray] = None) -> dict:
        if lifting_target is None or keypoints_3d is None:
            raise ValueError('Temporal lifting requires center and sequence 3D labels')
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # x is centred by width/2, y by height/2; both use width/2 scale,
        # exactly the convention used by MMPose's VideoPoseLifting codec.
        center = self.image_size * .5
        scale = self.image_size[0] * .5
        keypoint_labels = (keypoints.astype(np.float32) - center) / scale
        if self.concat_vis:
            keypoint_labels = np.concatenate(
                [keypoint_labels, keypoints_visible[..., None]], axis=-1)
        if self.concat_mask:
            if input_occlusion_mask is None:
                input_occlusion_mask = np.ones(
                    keypoints.shape[:2], dtype=np.float32)
            input_occlusion_mask = np.asarray(
                input_occlusion_mask, dtype=np.float32)
            if input_occlusion_mask.shape != keypoints.shape[:2]:
                raise ValueError(
                    'input_occlusion_mask must match (T, K), got '
                    f'{input_occlusion_mask.shape} for {keypoints.shape[:2]}')
            keypoint_labels = np.concatenate(
                [keypoint_labels, input_occlusion_mask[..., None]], axis=-1)
        keypoint_labels = keypoint_labels.transpose(1, 2, 0).reshape(
            -1, keypoint_labels.shape[0])

        single_label, root = self._root_relative(
            np.asarray(lifting_target, dtype=np.float32))
        sequence_label, _ = self._root_relative(
            np.asarray(keypoints_3d, dtype=np.float32))
        single_weight = self._weights(lifting_target_visible, lifting_target)
        sequence_weight = self._weights(keypoints_3d_visible, keypoints_3d)

        encoded = dict(
            keypoint_labels=keypoint_labels,
            keypoint_labels_visible=keypoints_visible,
            lifting_target_label=single_label,
            lifting_target_weight=single_weight,
            # InstanceData requires every label field to share its first
            # (instance) dimension. One temporal window is one instance.
            lifting_sequence_label=sequence_label[None, ...],
            lifting_sequence_weight=sequence_weight[None, ...],
            trajectory_weights=single_weight,
            target_root=root)
        if self.remove_root:
            encoded['target_root_removed'] = True
            if self.save_index:
                encoded['target_root_index'] = self.root_index[0]
        return encoded

    def decode(self,
               encoded: np.ndarray,
               target_root: Optional[np.ndarray] = None):
        keypoints = encoded.copy()
        if target_root is not None and target_root.size:
            # The network predicts root-relative joints. Restore the camera
            # coordinate frame for every non-root joint before inserting the
            # removed root itself.
            keypoints = keypoints + target_root
        if self.remove_root:
            root = (target_root if target_root is not None and target_root.size
                    else np.zeros((keypoints.shape[0], 3), dtype=keypoints.dtype))
            keypoints = np.insert(keypoints, self.root_index[0], root, axis=1)
        return keypoints, np.ones(keypoints.shape[:-1], dtype=np.float32)
