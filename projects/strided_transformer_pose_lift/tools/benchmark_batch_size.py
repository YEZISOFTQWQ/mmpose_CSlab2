#!/usr/bin/env python3
"""Measure realistic temporal pose-lifting throughput for batch sizes.

The script never writes model checkpoints. Each candidate receives a fresh
model and optimizer, then measures actual data loading, forward, backward and
AdamW update time after warm-up. It is intended to choose a batch size before
starting a new full training run.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader

import mmpose.datasets  # Register stock datasets and transforms.
from mmpose.registry import DATASETS, MODELS


def pose_collate(samples):
    """Match the packed 3D-pose input format used by the training runner."""
    return dict(inputs=torch.stack([sample['inputs'] for sample in samples]),
                data_samples=[sample['data_samples'] for sample in samples])


def move_samples(samples, device):
    return [sample.to(device) for sample in samples]


def train_step(model, optimizer, scaler, batch, device, amp):
    inputs = batch['inputs'].to(device, non_blocking=True)
    samples = move_samples(batch['data_samples'], device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        losses = model.loss(inputs, samples)
        loss = sum(value for name, value in losses.items()
                   if name.startswith('loss'))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def benchmark(dataset, cfg, batch_size, workers, warmup, iterations, amp):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=pose_collate)
    device = torch.device('cuda:0')
    model = MODELS.build(cfg.model).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim_wrapper.optimizer.lr,
        weight_decay=cfg.optim_wrapper.optimizer.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        train_step(model, optimizer, scaler, batch, device, amp)
    torch.cuda.synchronize(device)

    data_times, step_times = [], []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        train_step(model, optimizer, scaler, batch, device, amp)
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - start + data_times[-1])

    mean_step = sum(step_times) / len(step_times)
    result = dict(
        batch_size=batch_size,
        mean_step_seconds=round(mean_step, 5),
        mean_data_seconds=round(sum(data_times) / len(data_times), 5),
        samples_per_second=round(batch_size / mean_step, 2),
        peak_memory_mib=round(torch.cuda.max_memory_allocated(device) / 2**20,
                              1))
    del iterator, loader, optimizer, model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[128, 256, 384, 512])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output-dir', default='work_dirs/batch_size_benchmark')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark')

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    cfg = Config.fromfile(args.config)
    init_default_scope('mmpose')
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    results = []
    for batch_size in args.batch_sizes:
        try:
            result = benchmark(dataset, cfg, batch_size, args.workers,
                               args.warmup, args.iterations, args.amp)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = dict(batch_size=batch_size, status='cuda_oom')
            results.append(result)
            print(json.dumps(result), flush=True)

    valid = [item for item in results if 'samples_per_second' in item]
    best = max(valid, key=lambda item: item['samples_per_second']) if valid else None
    report = dict(
        timestamp=datetime.now().isoformat(timespec='seconds'),
        config=args.config,
        amp=args.amp,
        workers=args.workers,
        warmup=args.warmup,
        iterations=args.iterations,
        results=results,
        recommendation=best)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'batch_size_{datetime.now():%Y%m%d_%H%M%S}.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {output}')
    if best:
        print('Recommended batch size:', best['batch_size'])


if __name__ == '__main__':
    main()
