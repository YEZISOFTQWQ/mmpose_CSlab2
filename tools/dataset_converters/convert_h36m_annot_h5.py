#!/usr/bin/env python3
"""Convert legacy H36M HDF5 keypoint pairs to MMPose pose-lift NPZ files.

``part`` contains (N, 17, 2) image-space 2D points and ``S`` contains
(N, 17, 3) camera-space 3D points in millimetres. No RGB image is extracted
or read; image-name lists are retained only as stable sample identifiers.
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np

SPLITS = {'train': ('train.h5', 'train_images.txt'),
          'test': ('valid.h5', 'valid_images.txt')}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    return parser.parse_args()


def convert_split(annotation_dir, split, output_dir):
    h5_name, names_name = SPLITS[split]
    with h5py.File(annotation_dir / h5_name, 'r') as source:
        points_2d = source['part'][:].astype(np.float32)
        points_3d = source['S'][:].astype(np.float32) / 1000.0
        centers = source['center'][:].astype(np.float32)
        scales = source['scale'][:].astype(np.float32)
    names = np.asarray((annotation_dir / names_name).read_text(
        encoding='utf-8').splitlines())
    n = len(names)
    if (points_2d.shape != (n, 17, 2) or points_3d.shape != (n, 17, 3)
            or len(centers) != n or len(scales) != n):
        raise ValueError(f'{split}: inconsistent H36M annotation shapes')
    # BaseMocapDataset expects a trailing per-joint visibility channel.
    part = np.concatenate((points_2d, np.ones((n, 17, 1), np.float32)), -1)
    pose3d = np.concatenate((points_3d, np.ones((n, 17, 1), np.float32)), -1)
    destination = output_dir / f'h36m_{split}_keypoints.npz'
    np.savez_compressed(destination, imgname=names, center=centers,
                        scale=scales, part=part, S=pose3d)
    print(f'{split}: {n} paired samples -> {destination}')


def main():
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    output_dir = args.output_root / 'annotation_body3d' / 'fps10'
    output_dir.mkdir(parents=True, exist_ok=True)
    members = [f'h36m/annot/{name}' for pair in SPLITS.values() for name in pair]
    with tempfile.TemporaryDirectory(prefix='h36m_annot_') as temp_dir:
        with tarfile.open(args.archive) as archive:
            archive.extractall(temp_dir,
                               members=[archive.getmember(m) for m in members])
        annotation_dir = Path(temp_dir) / 'h36m' / 'annot'
        for split in SPLITS:
            convert_split(annotation_dir, split, output_dir)


if __name__ == '__main__':
    main()
