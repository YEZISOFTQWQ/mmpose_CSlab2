#!/usr/bin/env python3
"""Read-only readiness checks for the Strided Transformer experiment."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def video_key(name: str):
    subject, rest = Path(str(name)).name.split('_', 1)
    action, rest = rest.split('.', 1)
    camera, _ = rest.split('_', 1)
    return subject, action, camera


def inspect_split(root: Path, split: str, seq_len: int, pad: bool):
    ann_path = root / f'annotation_body3d/fps10/h36m_{split}_keypoints.npz'
    det_path = root / (
        f'annotation_body3d/fps10/rtmdet_rtmpose_m_{split}_h36m17.npy')
    missing = [str(path) for path in (ann_path, det_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    ann = np.load(ann_path, mmap_mode='r')
    det = np.load(det_path, mmap_mode='r')
    required = {'imgname', 'part', 'S'}
    missing_keys = required - set(ann.files)
    if missing_keys:
        raise ValueError(f'{ann_path}: missing NPZ keys {sorted(missing_keys)}')
    n = len(ann['imgname'])
    if det.shape != (n, 17, 3):
        raise ValueError(f'{det_path}: expected {(n, 17, 3)}, got {det.shape}')
    if ann['S'].shape != (n, 17, 4):
        raise ValueError(f'{ann_path}: unexpected 3D shape {ann["S"].shape}')
    if not np.isfinite(det).all() or not np.isfinite(ann['S']).all():
        raise ValueError(f'{split}: non-finite detection or 3D values')

    videos = defaultdict(int)
    for name in ann['imgname']:
        videos[video_key(name)] += 1
    half = (seq_len - 1) // 2
    if pad:
        windows = n
    else:
        windows = sum(max(0, frames - 2 * half) for frames in videos.values())
    print(f'{split}: frames={n:,}; videos={len(videos):,}; '
          f'windows(seq_len={seq_len}, pad={pad})={windows:,}; '
          f'detection={det.shape}')
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/h36m')
    parser.add_argument('--seq-len', type=int, default=9)
    args = parser.parse_args()
    if args.seq_len < 3 or args.seq_len % 2 != 1:
        raise ValueError('--seq-len must be an odd integer >= 3')
    root = Path(args.data_root)
    train = inspect_split(root, 'train', args.seq_len, pad=False)
    test = inspect_split(root, 'test', args.seq_len, pad=True)
    print(f'Ready: {train:,} train windows and {test:,} validation windows.')


if __name__ == '__main__':
    main()
