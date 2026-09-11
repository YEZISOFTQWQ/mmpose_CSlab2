#!/usr/bin/env python3
"""Run the desktop batch image/video RTMPose-to-3D inference application."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QStatusBar

from config import ModelPaths
from ui.batch_panel import BatchPanel


class PoseDesktopWindow(QMainWindow):
    """A batch-only GUI; camera access is intentionally out of scope."""

    def __init__(self, paths: ModelPaths, device: str):
        super().__init__()
        self.setWindowTitle('MMpose App CSlab')
        self.resize(920, 700)
        self.batch_panel = BatchPanel(paths, device, self)
        self.batch_panel.activity_changed.connect(self._set_activity)
        self.setCentralWidget(self.batch_panel)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Ready — add images, videos, or folders')

    def _set_activity(self, active):
        self.statusBar().showMessage('Batch GPU inference is running' if active else 'Ready')

    def closeEvent(self, event):
        self.batch_panel.stop()
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--det-checkpoint')
    parser.add_argument('--pose2d-checkpoint')
    parser.add_argument('--lifter-checkpoint',
                        help='Legacy alias for --single-lifter-checkpoint')
    parser.add_argument('--single-lifter-checkpoint')
    parser.add_argument('--temporal-lifter-checkpoint')
    parser.add_argument('--occlusion-lifter-checkpoint',
                        help='Override the v4 confidence/occlusion checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ModelPaths.defaults()
    single_checkpoint = (args.single_lifter_checkpoint or
                         args.lifter_checkpoint)
    if any((args.det_checkpoint, args.pose2d_checkpoint, single_checkpoint,
            args.temporal_lifter_checkpoint, args.occlusion_lifter_checkpoint)):
        paths = ModelPaths(
            paths.det_config,
            Path(args.det_checkpoint) if args.det_checkpoint else paths.det_checkpoint,
            paths.pose2d_config,
            Path(args.pose2d_checkpoint) if args.pose2d_checkpoint else paths.pose2d_checkpoint,
            paths.single_lifter_config,
            Path(single_checkpoint) if single_checkpoint else paths.single_lifter_checkpoint,
            paths.temporal_lifter_config,
            (Path(args.temporal_lifter_checkpoint)
             if args.temporal_lifter_checkpoint
             else paths.temporal_lifter_checkpoint),
            paths.occlusion_lifter_config,
            (Path(args.occlusion_lifter_checkpoint)
             if args.occlusion_lifter_checkpoint
             else paths.occlusion_lifter_checkpoint))
    app = QApplication(sys.argv)
    window = PoseDesktopWindow(paths, args.device)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
