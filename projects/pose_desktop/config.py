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
    lifter_config: Path
    lifter_checkpoint: Path

    @classmethod
    def defaults(cls) -> 'ModelPaths':
        resource_root = _resource_root()
        bundled = getattr(sys, 'frozen', False)
        assets = resource_root / 'models' if bundled else SOURCE_ROOT / 'artifacts/models'
        lifter_checkpoint = (assets / 'best_MPJPE_epoch_75.pth' if bundled else
                             SOURCE_ROOT / 'work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/'
                             'best_MPJPE_epoch_75.pth')
        return cls(
            det_config=resource_root / 'demo/mmdetection_cfg/'
            'rtmdet_m_640-8xb32_coco-person.py',
            det_checkpoint=assets /
            'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
            pose2d_config=resource_root / 'configs/body_2d_keypoint/rtmpose/'
            'body8/rtmpose-m_8xb256-420e_body8-256x192.py',
            pose2d_checkpoint=assets /
            'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
            lifter_config=resource_root / 'projects/single_image_pose_lift/'
            'image_pose_lift_tcn_h36m_rtmpose_v2.py',
            lifter_checkpoint=lifter_checkpoint)


@dataclass(frozen=True)
class RuntimeOptions:
    camera_index: int = 0
    device: str = 'cuda:0'
    bbox_threshold: float = 0.3
    max_people: int = 1
    norm_pose_2d: bool = False
