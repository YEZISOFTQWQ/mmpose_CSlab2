"""Memory-bounded temporal H36M dataset for the Strided Transformer run."""

from typing import List, Optional

import numpy as np
from mmengine.fileio import exists, get_local_path

from mmpose.datasets.datasets.body3d.h36m_dataset import Human36mDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class LazyHuman36mDataset(Human36mDataset):
    """Assemble H36M temporal windows when sampled, not at startup.

    ``Human36mDataset`` eagerly copies all 2D and 3D arrays into one Python
    dictionary per window. With 307k nine-frame windows that exhausts a small
    WSL VM before training begins. This variant serializes only empty sample
    placeholders and retrieves the nine rows required for a batch on demand.

    It intentionally supports the top-down detection-input path used by this
    project. It preserves H36M's original subject/action/camera window split.
    """

    def __init__(self,
                 *args,
                 keypoint_2d_src: str = 'detection',
                 data_mode: str = 'topdown',
                 factor_file: Optional[str] = None,
                 camera_param_file: Optional[str] = None,
                 **kwargs):
        if keypoint_2d_src != 'detection':
            raise ValueError('LazyHuman36mDataset requires detection 2D input')
        if data_mode != 'topdown':
            raise ValueError('LazyHuman36mDataset supports topdown mode only')
        if factor_file or camera_param_file:
            raise ValueError('Factors/camera parameters are unsupported here')
        super().__init__(
            *args,
            keypoint_2d_src=keypoint_2d_src,
            data_mode=data_mode,
            factor_file=None,
            camera_param_file=None,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Keep annotations in arrays; serialize only one tiny item per window."""
        if not hasattr(self, 'keypoint_2d_det_file'):
            raise ValueError('keypoint_2d_det_file is required')
        if not exists(self.keypoint_2d_det_file):
            raise FileNotFoundError(self.keypoint_2d_det_file)

        # NPZ fields are read once and shared copy-on-write by fork workers.
        self._img_names = np.asarray(self.ann_data['imgname'])
        self._keypoints_3d = np.asarray(self.ann_data['S'], dtype=np.float32)
        if self._keypoints_3d.ndim != 3 or self._keypoints_3d.shape[1:] != (17, 4):
            raise ValueError(f'Unexpected H36M 3D shape {self._keypoints_3d.shape}')
        # Detection input stays memory-mapped instead of occupying RAM.
        with get_local_path(self.keypoint_2d_det_file) as local_path:
            self._keypoints_2d = np.load(local_path, mmap_mode='r')
        expected_2d_shape = (*self._keypoints_3d.shape[:2], 3)
        if self._keypoints_2d.shape != expected_2d_shape:
            raise ValueError(
                'Unexpected 2D detection shape; got '
                f'{self._keypoints_2d.shape}, expected {expected_2d_shape}')

        # Convert the temporary list-of-lists made by Human36mDataset into a
        # compact int32 matrix: 307k x 9 uses about 11 MB rather than hundreds
        # of MB of Python integer/list objects.
        self.sequence_indices = np.asarray(self.sequence_indices, dtype=np.int32)
        return [{} for _ in range(len(self.sequence_indices))]

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        sequence_idx = data_info['sample_idx']
        frame_ids = self.sequence_indices[sequence_idx]
        keypoints_2d = np.asarray(self._keypoints_2d[frame_ids], dtype=np.float32)
        keypoints_3d = self._keypoints_3d[frame_ids]
        target_idx = -1 if self.causal else len(frame_ids) // 2
        img_names = self._img_names[frame_ids]

        data_info.update(
            num_keypoints=17,
            keypoints=keypoints_2d[..., :2],
            keypoints_visible=keypoints_2d[..., 2],
            keypoints_3d=keypoints_3d[..., :3],
            keypoints_3d_visible=keypoints_3d[..., 3],
            id=int(sequence_idx),
            category_id=1,
            iscrowd=0,
            img_paths=list(img_names),
            img_ids=frame_ids,
            lifting_target=keypoints_3d[target_idx:target_idx + 1, ..., :3],
            lifting_target_visible=keypoints_3d[
                target_idx:target_idx + 1, ..., 3],
            target_img_path=img_names[target_idx:target_idx + 1])
        return data_info
