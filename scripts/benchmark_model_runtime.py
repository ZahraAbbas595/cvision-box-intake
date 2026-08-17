"""Benchmark deployment-model memory and CPU latency on evaluation images."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse model, image, repeat, and runtime benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "carton_yolov8n_truck_best.pt",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "eval" / "dataset" / "images",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def rss_mb() -> float:
    """Return current process resident memory in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile from a non-empty value list."""
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    """Benchmark detector latency and memory under repeatable settings."""
    args = parse_args()
    images = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No evaluation images found in {args.images}")

    torch.set_num_threads(1)
    before_load_mb = rss_mb()
    load_started = time.perf_counter()
    model = YOLO(args.model)
    load_seconds = time.perf_counter() - load_started
    after_load_mb = rss_mb()

    latencies: list[float] = []
    peak_mb = after_load_mb
    for image_path in images:
        started = time.perf_counter()
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device="cpu",
            verbose=False,
        )
        latencies.append(time.perf_counter() - started)
        peak_mb = max(peak_mb, rss_mb())

    warm = latencies[1:] if len(latencies) > 1 else latencies
    print(f"images={len(images)}")
    print(f"rss_before_load_mb={before_load_mb:.1f}")
    print(f"rss_after_load_mb={after_load_mb:.1f}")
    print(f"rss_peak_mb={peak_mb:.1f}")
    print(f"model_load_seconds={load_seconds:.4f}")
    print(f"cold_inference_seconds={latencies[0]:.4f}")
    print(f"warm_mean_seconds={statistics.fmean(warm):.4f}")
    print(f"warm_median_seconds={statistics.median(warm):.4f}")
    print(f"warm_p95_seconds={percentile(warm, 0.95):.4f}")
    print(f"warm_max_seconds={max(warm):.4f}")


if __name__ == "__main__":
    main()
