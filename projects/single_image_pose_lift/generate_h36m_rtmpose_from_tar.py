#!/usr/bin/env python3
"""Create aligned H36M 2D detections without extracting RGB image archives.

For every row in an H36M paired NPZ, this script reads only that JPEG from the
appropriate ``S*.tar`` archive, runs the deployment detector and RTMPose, and
writes an aligned ``(N, 17, 3)`` float32 NPY array.  The three channels are
``x, y, confidence`` in the H36M-17 keypoint convention.

It checkpoints after each subject.  Re-running with ``--resume`` never
changes completed subjects and needs no extracted RGB dataset on disk.
"""

import argparse
import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import get_test_pipeline_cfg

from mmpose.apis import convert_keypoint_definition, inference_topdown, init_model
from mmpose.utils import adapt_mmdet_pipeline, register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', required=True, type=Path)
    parser.add_argument('--archive-dir', required=True, type=Path)
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--pose-config', required=True)
    parser.add_argument('--pose-checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bbox-thr', type=float, default=.3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def get_subject(name):
    return name.split('_', 1)[0]


def gt_bbox(center, scale):
    """Fallback only when RTMDet misses the known H36M person."""
    size = float(scale) * 200.
    return np.array([[center[0] - size / 2, center[1] - size / 2,
                      center[0] + size / 2, center[1] + size / 2]],
                    dtype=np.float32)


def read_state(path):
    if not path.is_file():
        return {'completed_subjects': [], 'subject_progress': {}}
    return json.loads(path.read_text(encoding='utf-8'))


def write_state(path, completed, subject_progress, total, failures):
    path.write_text(json.dumps({
        'completed_subjects': sorted(completed),
        'subject_progress': subject_progress,
        'total_samples': total,
        'detector_fallbacks': failures,
    }, indent=2), encoding='utf-8')


def infer_batch(images, indices, detector, detector_pipeline, pose_model,
                pose_pipeline, centers, scales, bbox_thr):
    """Run detector and pose model once per batch, not once per image."""
    det_data = [detector_pipeline(dict(img=image, img_id=i))
                for i, image in enumerate(images)]
    det_batch = dict(inputs=[item['inputs'] for item in det_data],
                     data_samples=[item['data_samples'] for item in det_data])
    with torch.no_grad():
        det_results = detector.test_step(det_batch)

    bboxes, fallback_positions = [], []
    for position, (sample, index) in enumerate(zip(det_results, indices)):
        pred = sample.pred_instances.cpu().numpy()
        people = np.flatnonzero((pred.labels == 0) & (pred.scores >= bbox_thr))
        if len(people):
            person = people[np.argmax(pred.scores[people])]
            bboxes.append(pred.bboxes[person:person + 1])
        else:
            bboxes.append(gt_bbox(centers[index], scales[index]))
            fallback_positions.append(position)

    pose_data = []
    for image, bbox in zip(images, bboxes):
        item = dict(img=image, bbox=bbox, bbox_score=np.ones(1, np.float32))
        item.update(pose_model.dataset_meta)
        pose_data.append(pose_pipeline(item))
    with torch.no_grad():
        pose_results = pose_model.test_step(pseudo_collate(pose_data))

    output = []
    for pose in pose_results:
        pred = pose.pred_instances.cpu().numpy()
        coco = np.concatenate((pred.keypoints[0], pred.keypoint_scores[0, :, None]),
                              axis=-1)[None]
        output.append(convert_keypoint_definition(coco, 'coco', 'h36m')[0])
    return np.asarray(output, dtype=np.float32), fallback_positions


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output.with_suffix(args.output.suffix + '.progress.json')
    failures_path = args.output.with_suffix(args.output.suffix + '.fallbacks.csv')
    with np.load(args.ann_file) as annotations:
        names = annotations['imgname'].astype(str)
        centers = annotations['center'].astype(np.float32)
        scales = annotations['scale'].astype(np.float32)
    groups = defaultdict(list)
    for index, name in enumerate(names):
        groups[get_subject(name)].append(index)

    if args.output.exists() and args.resume:
        detections = np.lib.format.open_memmap(args.output, mode='r+')
        if detections.shape != (len(names), 17, 3):
            raise ValueError(f'Unexpected resume shape: {detections.shape}')
        state = read_state(state_path)
    elif args.output.exists():
        raise FileExistsError(f'{args.output} exists; use --resume to continue')
    else:
        detections = np.lib.format.open_memmap(
            args.output, mode='w+', dtype=np.float32, shape=(len(names), 17, 3))
        state = {'completed_subjects': []}
    completed = set(state['completed_subjects'])
    subject_progress = dict(state.get('subject_progress', {}))
    fallback_count = int(state.get('detector_fallbacks', 0))

    register_all_modules()
    detector = init_detector(args.det_config, args.det_checkpoint, args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_model = init_model(args.pose_config, args.pose_checkpoint, args.device)
    det_pipeline_cfg = get_test_pipeline_cfg(detector.cfg.copy())
    det_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    detector_pipeline = Compose(det_pipeline_cfg)
    pose_pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)

    new_file = not failures_path.exists() or not args.resume
    with failures_path.open('a', newline='', encoding='utf-8') as failure_file:
        writer = csv.writer(failure_file)
        if new_file:
            writer.writerow(['index', 'image', 'reason'])
        for subject, indices in sorted(groups.items()):
            if subject in completed:
                print(f'Skip completed {subject}: {len(indices)} samples', flush=True)
                continue
            archive_path = args.archive_dir / f'{subject}.tar'
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            # Old interrupted runs may predate the progress journal. Infer a
            # contiguous completed prefix from the nonzero keypoint rows.
            start_at = int(subject_progress.get(subject, 0))
            if start_at == 0:
                written = np.any(detections[indices] != 0, axis=(1, 2))
                unfinished = np.flatnonzero(~written)
                start_at = int(unfinished[0]) if len(unfinished) else len(indices)
                if start_at:
                    print(f'Resume {subject} at {start_at}/{len(indices)}', flush=True)
            if start_at == len(indices):
                completed.add(subject)
                subject_progress[subject] = len(indices)
                write_state(state_path, completed, subject_progress, len(names),
                            fallback_count)
                continue
            print(f'Process {subject}: {start_at}/{len(indices)} done from {archive_path}', flush=True)
            with tarfile.open(archive_path) as archive:
                members = {member.name: member for member in archive.getmembers()}
                for start in range(start_at, len(indices), args.batch_size):
                    batch_indices = indices[start:start + args.batch_size]
                    images = []
                    for index in batch_indices:
                        name = names[index]
                        member = members.get(name)
                        if member is None:
                            raise FileNotFoundError(f'{name} missing from {archive_path}')
                        encoded = archive.extractfile(member).read()
                        image = cv2.imdecode(np.frombuffer(encoded, np.uint8),
                                             cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError(f'Cannot decode {name}')
                        images.append(image)
                    result, fallback_positions = infer_batch(
                        images, batch_indices, detector, detector_pipeline,
                        pose_model, pose_pipeline, centers, scales, args.bbox_thr)
                    detections[batch_indices] = result
                    for position in fallback_positions:
                        index = batch_indices[position]
                        fallback_count += 1
                        writer.writerow([index, names[index], 'rtmdet_person_missing'])
                    local_count = start + len(batch_indices)
                    subject_progress[subject] = local_count
                    # The journal is tiny. Persist every batch so an abrupt
                    # shutdown resumes within at most one 16-image batch.
                    write_state(state_path, completed, subject_progress,
                                len(names), fallback_count)
                    if local_count % 1000 < args.batch_size or local_count == len(indices):
                        detections.flush()
                        print(f'{subject}: {local_count}/{len(indices)}', flush=True)
            completed.add(subject)
            detections.flush()
            subject_progress[subject] = len(indices)
            write_state(state_path, completed, subject_progress, len(names),
                        fallback_count)
            print(f'Completed {subject}; fallbacks so far: {fallback_count}', flush=True)
    print(f'Saved {detections.shape} to {args.output}', flush=True)
    print(f'Fallback records: {failures_path}', flush=True)


if __name__ == '__main__':
    main()
