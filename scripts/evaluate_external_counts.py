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
    within_one: int = 0
    within_two: int = 0
    overcounted: int = 0
    undercounted: int = 0

    def add(self, ground_truth: int, predicted: int) -> None:
        """Add one image result."""
        error = abs(predicted - ground_truth)
        self.images += 1
        self.ground_truth += ground_truth
        self.predicted += predicted
        self.absolute_error += error
        self.exact += int(error == 0)
        self.within_one += int(error <= 1)
        self.within_two += int(error <= 2)
        self.overcounted += int(predicted > ground_truth)
        self.undercounted += int(predicted < ground_truth)


@dataclass(frozen=True)
class ImageCountResult:
    """Store one image's count outcome for reproducible error examples."""

    image: str
    ground_truth: int
    predicted: int

    @property
    def signed_error(self) -> int:
        """Return predicted minus ground-truth count."""
        return self.predicted - self.ground_truth


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, and inference settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "runs/detect/carton_counter_truck/weights/best.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "external_test/carton_loading_evaluation",
    )
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument(
        "--example-count",
        type=int,
        default=3,
        help="Number of exact, overcount, and undercount examples to print",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference and print Carton Loading count metrics."""
    args = parse_args()
    metrics = CountMetrics()
    image_results: list[ImageCountResult] = []
    yolo_class: Any = import_module("ultralytics").YOLO
    model = yolo_class(args.model)
    results = model.predict(
        source=str(args.dataset / "images"),
        imgsz=640,
        conf=args.confidence,
        iou=args.iou,
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
        image_results.append(ImageCountResult(stem, ground_truth, predicted))
        metrics.add(ground_truth, predicted)

    mean_absolute_error = metrics.absolute_error / metrics.images
    exact_accuracy = 100 * metrics.exact / metrics.images
    within_one_accuracy = 100 * metrics.within_one / metrics.images
    within_two_accuracy = 100 * metrics.within_two / metrics.images
    print(
        f"carton_loading: images={metrics.images} "
        f"ground_truth={metrics.ground_truth} predicted={metrics.predicted} "
        f"MAE={mean_absolute_error:.2f} exact={exact_accuracy:.1f}% "
        f"within_one={within_one_accuracy:.1f}% "
        f"within_two={within_two_accuracy:.1f}% "
        f"overcounted={metrics.overcounted} undercounted={metrics.undercounted}"
    )

    exact_examples = [item for item in image_results if item.signed_error == 0]
    overcount_examples = sorted(
        (item for item in image_results if item.signed_error > 0),
        key=lambda item: item.signed_error,
        reverse=True,
    )
    undercount_examples = sorted(
        (item for item in image_results if item.signed_error < 0),
        key=lambda item: item.signed_error,
    )
    for category, examples in (
        ("exact", exact_examples),
        ("overcount", overcount_examples),
        ("undercount", undercount_examples),
    ):
        for item in examples[: args.example_count]:
            print(
                f"example={category} image={item.image} "
                f"ground_truth={item.ground_truth} predicted={item.predicted} "
                f"signed_error={item.signed_error:+d}"
            )


if __name__ == "__main__":
    main()
