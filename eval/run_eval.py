"""Evaluate operational box counts on the curated evaluation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
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
        default=ROOT / "eval" / "results" / "count_metrics.csv",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    args = parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    model = YOLO(args.model)
    rows: list[dict[str, str | int | float]] = []

    for filename, expected in ground_truth["images"].items():
        image_path = args.images / filename
        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device="cpu",
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        predicted_count = len(result.boxes)
        expected_count = int(expected["expected_visible_count"])
        error = predicted_count - expected_count
        rows.append(
            {
                "image": filename,
                "expected_count": expected_count,
                "predicted_count": predicted_count,
                "signed_error": error,
                "absolute_error": abs(error),
                "outcome": (
                    "exact"
                    if error == 0
                    else ("overcount" if error > 0 else "undercount")
                ),
                "inference_ms": round(elapsed_ms, 2),
                "ambiguity": expected.get("ambiguity", ""),
                "notes": expected.get("notes", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    absolute_errors = [int(row["absolute_error"]) for row in rows]
    inference_times = [float(row["inference_ms"]) for row in rows]
    exact = sum(row["outcome"] == "exact" for row in rows)
    overcounts = sum(row["outcome"] == "overcount" for row in rows)
    undercounts = sum(row["outcome"] == "undercount" for row in rows)
    print(f"images={len(rows)}")
    print(f"exact_count_accuracy={exact / len(rows):.4f}")
    print(f"mean_absolute_count_error={statistics.fmean(absolute_errors):.4f}")
    print(f"overcount_images={overcounts}")
    print(f"undercount_images={undercounts}")
    print(f"mean_inference_ms={statistics.fmean(inference_times):.2f}")
    print(f"p95_inference_ms={percentile(inference_times, 0.95):.2f}")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
