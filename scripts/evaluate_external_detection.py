"""Measure box-level metrics on the Carton Loading evaluation set."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, and deployed inference settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/carton_yolov8n_truck_best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "external_test/carton_loading_evaluation/data.yaml",
    )
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    """Run IoU-matched YOLO validation and print box-level metrics."""
    args = parse_args()
    yolo_class: Any = import_module("ultralytics").YOLO
    model = yolo_class(args.model)
    results: Any = model.val(
        data=str(args.data),
        split="val",
        imgsz=640,
        conf=args.confidence,
        iou=args.iou,
        max_det=300,
        device="cpu",
        plots=False,
        save_json=False,
    )
    print(
        f"precision={results.box.mp:.4f} recall={results.box.mr:.4f} "
        f"mAP50={results.box.map50:.4f} mAP50-95={results.box.map:.4f}"
    )


if __name__ == "__main__":
    main()
