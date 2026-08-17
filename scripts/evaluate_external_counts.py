"""Measure per-image count accuracy on the normalized external test set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CountMetrics:
    """Accumulate operational carton-count metrics."""

    images: int = 0
    ground_truth: int = 0
    predicted: int = 0
    absolute_error: int = 0
    exact: int = 0

    def add(self, ground_truth: int, predicted: int) -> None:
        """Add one image result."""
        error = abs(predicted - ground_truth)
        self.images += 1
        self.ground_truth += ground_truth
        self.predicted += predicted
        self.absolute_error += error
        self.exact += int(error == 0)


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, and inference settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "runs/detect/carton_counter_truck/weights/best.pt",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "external_test/combined")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def source_name(stem: str) -> str:
    """Return the source name encoded by the dataset builder."""
    if stem.startswith("boxintake_"):
        return "boxintake"
    if stem.startswith("carton_loading_"):
        return "carton_loading"
    raise ValueError(f"Unknown source prefix: {stem}")


def main() -> None:
    """Run inference and print overall and per-source count accuracy."""
    args = parse_args()
    metrics = {
        "all": CountMetrics(),
        "boxintake": CountMetrics(),
        "carton_loading": CountMetrics(),
    }
    yolo_class: Any = import_module("ultralytics").YOLO
    model = yolo_class(args.model)
    results = model.predict(
        source=str(args.dataset / "images"),
        imgsz=640,
        conf=args.confidence,
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
    )
    for result in results:
        stem = Path(result.path).stem
        labels = args.dataset / "labels" / f"{stem}.txt"
        ground_truth = sum(
            bool(line.strip())
            for line in labels.read_text(encoding="utf-8").splitlines()
        )
        predicted = len(result.boxes)
        metrics["all"].add(ground_truth, predicted)
        metrics[source_name(stem)].add(ground_truth, predicted)

    for name, values in metrics.items():
        mean_absolute_error = values.absolute_error / values.images
        exact_accuracy = 100 * values.exact / values.images
        print(
            f"{name}: images={values.images} ground_truth={values.ground_truth} "
            f"predicted={values.predicted} MAE={mean_absolute_error:.2f} "
            f"exact={exact_accuracy:.1f}%"
        )


if __name__ == "__main__":
    main()
