"""Sweep confidence and NMS IoU thresholds using operational count metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse PyTorch model, reviewed data, and report settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "carton_yolov8n_best.pt",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "eval" / "dataset" / "ground_truth.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "eval" / "dataset" / "images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "results" / "threshold_sweep.csv",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    """Sweep confidence and IoU settings for the PyTorch detector."""
    args = parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    filenames = list(ground_truth["images"])
    image_paths = [str(args.images / filename) for filename in filenames]
    expected_counts = [
        int(ground_truth["images"][filename]["expected_visible_count"])
        for filename in filenames
    ]
    model = YOLO(args.model)

    raw_predictions: list[tuple[torch.Tensor, torch.Tensor]] = []
    for image_path in image_paths:
        result = model.predict(
            source=image_path,
            conf=0.20,
            iou=0.70,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        raw_predictions.append(
            (
                result.boxes.xyxy.detach().cpu(),
                result.boxes.conf.detach().cpu(),
            )
        )

    rows: list[dict[str, float | int]] = []
    for confidence_step in range(20, 61, 5):
        confidence = confidence_step / 100
        for iou_step in range(30, 71, 10):
            iou = iou_step / 100
            predicted_counts: list[int] = []
            for boxes, scores in raw_predictions:
                accepted = scores >= confidence
                filtered_boxes = boxes[accepted]
                filtered_scores = scores[accepted]
                kept = nms(filtered_boxes, filtered_scores, iou)
                predicted_counts.append(len(kept))
            errors = [
                predicted - expected
                for predicted, expected in zip(
                    predicted_counts,
                    expected_counts,
                    strict=True,
                )
            ]
            absolute_errors = [abs(error) for error in errors]
            exact = sum(error == 0 for error in errors)
            rows.append(
                {
                    "confidence": confidence,
                    "iou": iou,
                    "exact_count_accuracy": round(exact / len(errors), 4),
                    "mean_absolute_count_error": round(
                        statistics.fmean(absolute_errors),
                        4,
                    ),
                    "overcount_images": sum(error > 0 for error in errors),
                    "undercount_images": sum(error < 0 for error in errors),
                    "total_absolute_error": sum(absolute_errors),
                }
            )

    rows.sort(
        key=lambda row: (
            -float(row["exact_count_accuracy"]),
            float(row["mean_absolute_count_error"]),
            -float(row["confidence"]),
            float(row["iou"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Top threshold combinations:")
    for row in rows[:10]:
        print(row)
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
