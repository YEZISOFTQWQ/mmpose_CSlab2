"""Background worker for batch v2, v3 and v4 pose inference."""

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from config import ModelPaths, RuntimeOptions
from inference.pipeline import PosePipeline
from inference.rendering import render_prediction


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
VIDEO_SUFFIXES = {'.avi', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.webm'}


@dataclass(frozen=True)
class BatchOptions:
    inputs: tuple[Path, ...]
    output_root: Path
    show_keypoints: bool = True
    show_bbox: bool = True
    show_3d: bool = True
    save_keypoints: bool = True
    max_people: int = 1
    lifter_mode: str = 'single_frame_v2'

    @property
    def temporal(self) -> bool:
        return self.lifter_mode.startswith('temporal_')


def discover_media(inputs):
    """Return a deterministic de-duplicated set of images and videos."""
    found = []
    for entry in inputs:
        if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            found.append(entry)
        elif entry.is_dir():
            found.extend(path for path in entry.rglob('*')
                         if path.is_file() and path.suffix.lower() in
                         IMAGE_SUFFIXES | VIDEO_SUFFIXES)
    return sorted(dict.fromkeys(path.resolve() for path in found))


def prediction_record(predictions):
    return [{
        'bbox_xyxy': prediction.bbox.round(3).tolist(),
        'detector_score': round(prediction.score, 6),
        'keypoints_2d_h36m17': prediction.keypoints_2d.round(3).tolist(),
        'keypoints_3d_h36m17_root_relative': prediction.keypoints_3d.round(6).tolist(),
    } for prediction in predictions]


class BatchInferenceWorker(QThread):
    status = Signal(str)
    progress = Signal(int, int, str)
    completed = Signal(str, bool)
    failed = Signal(str)

    def __init__(self, paths: ModelPaths, device: str, options: BatchOptions):
        super().__init__()
        self.paths = paths
        self.device = device
        self.options = options
        self._running = False

    def stop(self):
        self._running = False

    def _write_image_output(self, frame, predictions, destination):
        output = render_prediction(frame, predictions, self.options.show_keypoints,
                                   self.options.show_bbox, self.options.show_3d)
        if not cv2.imwrite(str(destination), output):
            raise IOError('cannot write image output')

    def _process_image(self, pipeline, source, destination, keypoint_path):
        if self.options.temporal:
            raise ValueError('The v3/v4 9-frame models support videos only; '
                             'select v2 for image inference')
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError('cannot decode image')
        predictions = pipeline.predict(image)
        self._write_image_output(image, predictions, destination)
        if self.options.save_keypoints:
            keypoint_path.write_text(json.dumps({
                'source': str(source), 'lifter_mode': 'single_frame_v2',
                'predictions': prediction_record(predictions)},
                ensure_ascii=False, indent=2), encoding='utf-8')

    def _open_video_outputs(self, source, keypoint_path):
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError('cannot open video')
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        keypoint_file = keypoint_path.open('w', encoding='utf-8') \
            if self.options.save_keypoints else None
        return capture, fps, keypoint_file

    @staticmethod
    def _make_writer(destination, fps, output):
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (output.shape[1], output.shape[0]))
        if not writer.isOpened():
            raise IOError('cannot create output video')
        return writer

    def _process_video(self, pipeline, source, destination, keypoint_path):
        """Run v2 once per input frame."""
        capture, fps, keypoint_file = self._open_video_outputs(source, keypoint_path)
        writer = None
        frames = 0
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    break
                predictions = pipeline.predict(frame)
                output = render_prediction(frame, predictions,
                                           self.options.show_keypoints,
                                           self.options.show_bbox,
                                           self.options.show_3d)
                if writer is None:
                    writer = self._make_writer(destination, fps, output)
                writer.write(output)
                if keypoint_file:
                    keypoint_file.write(json.dumps({
                        'frame_index': frames, 'lifter_mode': 'single_frame_v2',
                        'predictions': prediction_record(predictions)},
                        ensure_ascii=False) + '\n')
                frames += 1
        finally:
            capture.release()
            if writer:
                writer.release()
            if keypoint_file:
                keypoint_file.close()
        return frames

    def _process_temporal_video(self, pipeline, source, destination,
                                keypoint_path):
        """Run v3/v4 with a bounded, padded 9-frame centred window.

        Every real frame is detected exactly once. Only nine `(frame,
        detections)` records are held at once. Repeated edge records implement
        the same boundary-padding intent as v3 validation data.
        """
        capture, fps, keypoint_file = self._open_video_outputs(source, keypoint_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            capture.release()
            if keypoint_file:
                keypoint_file.close()
            raise ValueError('video contains no readable frames')

        def read_detection():
            ok, frame = capture.read()
            return (frame, pipeline.detect_2d(frame)) if ok else None

        first = read_detection()
        if first is None:
            capture.release()
            if keypoint_file:
                keypoint_file.close()
            raise ValueError('cannot decode first video frame')
        window = deque([first] * 5, maxlen=9)
        for _ in range(4):
            following = read_detection()
            window.append(following if following is not None else window[-1])

        writer = None
        frames = 0
        try:
            for frame_index in range(total_frames):
                if not self._running:
                    break
                center_frame, _ = window[4]
                height, width = center_frame.shape[:2]
                predictions = pipeline.predict_temporal(
                    [detections for _, detections in window], (width, height))
                output = render_prediction(center_frame, predictions,
                                           self.options.show_keypoints,
                                           self.options.show_bbox,
                                           self.options.show_3d)
                if writer is None:
                    writer = self._make_writer(destination, fps, output)
                writer.write(output)
                if keypoint_file:
                    keypoint_file.write(json.dumps({
                        'frame_index': frame_index,
                        'lifter_mode': self.options.lifter_mode + '_noncausal_9frame',
                        'window_indices': [max(0, frame_index - 4), frame_index,
                                           min(total_frames - 1, frame_index + 4)],
                        'predictions': prediction_record(predictions)},
                        ensure_ascii=False) + '\n')
                frames += 1
                if frame_index + 1 < total_frames:
                    following = read_detection()
                    window.append(following if following is not None else window[-1])
        finally:
            capture.release()
            if writer:
                writer.release()
            if keypoint_file:
                keypoint_file.close()
        return frames

    def run(self):
        self._running = True
        media = discover_media(self.options.inputs)
        if not media:
            self.failed.emit('No supported image or video files were found')
            return
        run_dir = self.options.output_root / datetime.now().strftime('batch_%Y%m%d_%H%M%S')
        image_dir, video_dir, keypoint_dir = (run_dir / 'images', run_dir / 'videos',
                                              run_dir / 'keypoints')
        for directory in (image_dir, video_dir, keypoint_dir):
            directory.mkdir(parents=True, exist_ok=True)
        manifest = {'created_at': datetime.now().isoformat(timespec='seconds'),
                    'options': {'show_keypoints': self.options.show_keypoints,
                                'show_bbox': self.options.show_bbox,
                                'show_3d': self.options.show_3d,
                                'save_keypoints': self.options.save_keypoints,
                                'max_people': self.options.max_people,
                                'lifter_mode': self.options.lifter_mode},
                    'items': []}
        try:
            self.status.emit('Loading models on ' + self.device)
            pipeline = PosePipeline(self.paths, RuntimeOptions(
                device=self.device, max_people=self.options.max_people,
                lifter_mode=self.options.lifter_mode))
            pipeline.load()
            for number, source in enumerate(media, 1):
                if not self._running:
                    break
                stem = f'{number:04d}_{source.stem}_pose'
                self.progress.emit(number - 1, len(media), source.name)
                try:
                    if source.suffix.lower() in IMAGE_SUFFIXES:
                        output = image_dir / f'{stem}{source.suffix.lower()}'
                        keypoints = keypoint_dir / f'{stem}.json'
                        self._process_image(pipeline, source, output, keypoints)
                        item = {'source': str(source), 'type': 'image',
                                'output': str(output.relative_to(run_dir))}
                    else:
                        output = video_dir / f'{stem}.mp4'
                        keypoints = keypoint_dir / f'{stem}.jsonl'
                        process_video = (self._process_temporal_video
                                         if self.options.temporal else self._process_video)
                        frames = process_video(pipeline, source, output, keypoints)
                        item = {'source': str(source), 'type': 'video',
                                'frames_processed': frames,
                                'output': str(output.relative_to(run_dir))}
                    manifest['items'].append(item)
                except Exception as exc:
                    manifest['items'].append({'source': str(source), 'status': 'failed',
                                              'error': f'{type(exc).__name__}: {exc}'})
                    self.status.emit(f'Skipped {source.name}: {exc}')
                self.progress.emit(number, len(media), source.name)
            (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                indent=2), encoding='utf-8')
            self.completed.emit(str(run_dir), self._running)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
