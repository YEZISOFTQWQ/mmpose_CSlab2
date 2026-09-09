"""Centralized paths and runtime options for the desktop application."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _resource_root() -> Path:
    """Locate bundled configs when running from a PyInstaller release."""
    explicit_root = os.environ.get('MMPOSE_DESKTOP_RESOURCE_ROOT')
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return SOURCE_ROOT


@dataclass(frozen=True)
class ModelPaths:
    """Paths remain configurable because weights are deliberately untracked."""

    det_config: Path
    det_checkpoint: Path
    pose2d_config: Path
    pose2d_checkpoint: Path
    single_lifter_config: Path
    single_lifter_checkpoint: Path
    temporal_lifter_config: Path
    temporal_lifter_checkpoint: Path

    def lifter_paths(self, temporal: bool) -> tuple[Path, Path]:
        """Return the config/checkpoint pair for the selected GUI mode."""
        if temporal:
            return self.temporal_lifter_config, self.temporal_lifter_checkpoint
        return self.single_lifter_config, self.single_lifter_checkpoint

    @classmethod
    def defaults(cls) -> 'ModelPaths':
        resource_root = _resource_root()
        bundled = getattr(sys, 'frozen', False)
        assets = resource_root / 'models' if bundled else SOURCE_ROOT / 'artifacts/models'
        single_lifter_checkpoint = (
            assets / 'best_MPJPE_epoch_75.pth' if bundled else SOURCE_ROOT /
            'work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/'
            'best_MPJPE_epoch_75.pth')
        # v3 is a local H36M experiment and is deliberately not bundled.
        temporal_lifter_checkpoint = SOURCE_ROOT / \
            'work_dirs/strided_transformer_h36m_rtmpose_9frm/' \
            'best_MPJPE_epoch_70.pth'
        return cls(
            det_config=resource_root / 'demo/mmdetection_cfg/'
            'rtmdet_m_640-8xb32_coco-person.py',
            det_checkpoint=assets /
            'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
            pose2d_config=resource_root / 'configs/body_2d_keypoint/rtmpose/'
            'body8/rtmpose-m_8xb256-420e_body8-256x192.py',
            pose2d_checkpoint=assets /
            'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
            single_lifter_config=resource_root /
            'projects/single_image_pose_lift/'
            'image_pose_lift_tcn_h36m_rtmpose_v2.py',
            single_lifter_checkpoint=single_lifter_checkpoint,
            temporal_lifter_config=resource_root /
            'projects/strided_transformer_pose_lift/'
            'strided_transformer_h36m_rtmpose_9frm.py',
            temporal_lifter_checkpoint=temporal_lifter_checkpoint)


@dataclass(frozen=True)
class RuntimeOptions:
    camera_index: int = 0
    device: str = 'cuda:0'
    bbox_threshold: float = 0.3
    max_people: int = 1
    norm_pose_2d: bool = False
    temporal: bool = False
