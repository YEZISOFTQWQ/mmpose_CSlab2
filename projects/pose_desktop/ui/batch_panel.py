"""Batch image/video inference tab for the desktop application."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPushButton, QProgressBar, QSpinBox, QVBoxLayout,
                               QWidget)

from config import ModelPaths, SOURCE_ROOT
from workers.batch import BatchInferenceWorker, BatchOptions


class BatchPanel(QWidget):
    activity_changed = Signal(bool)

    def __init__(self, paths: ModelPaths, device: str, parent=None):
        super().__init__(parent)
        self.paths, self.device = paths, device
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        self.inputs = QListWidget()
        self.inputs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.output_root = QLineEdit(str(SOURCE_ROOT / 'artifacts/batch_output'))
        self.show_keypoints = QCheckBox('Draw 2D skeleton')
        self.show_bbox = QCheckBox('Draw person bounding box')
        self.show_3d = QCheckBox('Append 3D pose panel')
        self.save_keypoints = QCheckBox('Export H36M-17 keypoints JSON')
        for checkbox in (self.show_keypoints, self.show_bbox, self.show_3d,
                         self.save_keypoints):
            checkbox.setChecked(True)
        self.lifter_mode = QComboBox()
        self.lifter_mode.addItem('v2 single-frame model (images and videos)', False)
        self.lifter_mode.addItem('v3 9-frame temporal model (video, one person)', True)
        self.lifter_mode.currentIndexChanged.connect(self._update_mode_controls)
        self.max_people = QSpinBox()
        self.max_people.setRange(1, 4)
        self.max_people.setValue(1)
        self.temporal_note = QLabel(
            'v3 uses t-4…t+4 and repeats boundary frames. It processes the '
            'highest-score person only; images require v2.')
        self.temporal_note.setWordWrap(True)
        self.start_button = QPushButton('Run batch inference')
        self.cancel_button = QPushButton('Cancel after current frame')
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.status = QLabel('Add one or more images, videos, or folders.')

        add_files = QPushButton('Add files')
        add_folder = QPushButton('Add folder')
        remove = QPushButton('Remove selected')
        clear = QPushButton('Clear list')
        add_files.clicked.connect(self.add_files)
        add_folder.clicked.connect(self.add_folder)
        remove.clicked.connect(lambda: [self.inputs.takeItem(self.inputs.row(item))
                                        for item in self.inputs.selectedItems()])
        clear.clicked.connect(self.inputs.clear)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)

        input_actions = QHBoxLayout()
        for button in (add_files, add_folder, remove, clear):
            input_actions.addWidget(button)
        input_group = QGroupBox('Input media')
        input_layout = QVBoxLayout(input_group)
        input_layout.addWidget(QLabel(
            'Folders are scanned recursively for supported images and videos.'))
        input_layout.addWidget(self.inputs)
        input_layout.addLayout(input_actions)

        output_browse = QPushButton('Choose output root')
        output_browse.clicked.connect(self.choose_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_root)
        output_layout.addWidget(output_browse)
        options_group = QGroupBox('Output annotations')
        form = QFormLayout(options_group)
        form.addRow('Output root', output_layout)
        form.addRow(self.show_keypoints)
        form.addRow(self.show_bbox)
        form.addRow(self.show_3d)
        form.addRow(self.save_keypoints)
        form.addRow('3D model', self.lifter_mode)
        form.addRow('Maximum people per frame', self.max_people)
        form.addRow(self.temporal_note)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(input_group, 3)
        layout.addWidget(options_group)
        layout.addLayout(controls)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        self._update_mode_controls()

    def _update_mode_controls(self):
        temporal = bool(self.lifter_mode.currentData())
        self.max_people.setEnabled(not temporal)
        if temporal:
            self.max_people.setValue(1)
        self.temporal_note.setVisible(temporal)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select images or videos', '',
            'Media (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm);;All files (*)')
        self._add_paths(paths)

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select input folder')
        if path:
            self._add_paths([path])

    def _add_paths(self, paths):
        existing = {self.inputs.item(index).text() for index in range(self.inputs.count())}
        for path in paths:
            absolute = str(Path(path).resolve())
            if absolute not in existing:
                self.inputs.addItem(absolute)
                existing.add(absolute)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Select output root',
                                                self.output_root.text())
        if path:
            self.output_root.setText(path)

    def start(self):
        paths = tuple(Path(self.inputs.item(index).text())
                      for index in range(self.inputs.count()))
        if not paths:
            QMessageBox.warning(self, 'No input',
                                'Add at least one image, video, or folder.')
            return
        output_root = Path(self.output_root.text()).expanduser()
        temporal = bool(self.lifter_mode.currentData())
        options = BatchOptions(paths, output_root, self.show_keypoints.isChecked(),
                               self.show_bbox.isChecked(), self.show_3d.isChecked(),
                               self.save_keypoints.isChecked(), self.max_people.value(),
                               temporal)
        self.worker = BatchInferenceWorker(self.paths, self.device, options)
        self.worker.status.connect(self.status.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.complete)
        self.worker.failed.connect(self.fail)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.activity_changed.emit(True)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.stop()
            self.status.setText('Cancelling after the current frame...')
            self.cancel_button.setEnabled(False)

    def update_progress(self, current, total, name):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.status.setText(f'[{current}/{total}] {name}')

    def complete(self, output_dir, completed):
        prefix = 'Completed: ' if completed else 'Cancelled; partial results: '
        self.status.setText(prefix + output_dir)
        self._finish()

    def fail(self, message):
        self._finish()
        QMessageBox.critical(self, 'Batch inference error', message)

    def _finish(self):
        self.worker = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.activity_changed.emit(False)

    def stop(self):
        if self.worker:
            self.worker.stop()
