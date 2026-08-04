"""Analyze review-only count-risk flags against reviewed evaluation data."""

import argparse
import csv
import json
from pathlib import Path

import cv2

from app.config import (
    FRAGMENT_MAX_AREA_RATIO,
    FRAGMENT_MAX_DETECTIONS,
    FRAGMENT_MAX_GAP_RATIO,
    FRAGMENT_MAX_IOU,
    FRAGMENT_MIN_AXIS_OVERLAP,
    HIGH_DETECTION_COUNT,
)
from app.services.count_risk import find_suspicious_fragment_pairs
from app.services.detector import run_inference


def parse_args() -> argparse.Namespace:
    """Parse reviewed manifest, image directory, and report destination."""
    parser = argparse.ArgumentParser(description="Analyze count-risk review flags.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("eval/dataset/ground_truth.json"),
    )
    parser.add_argument("--images", type=Path, default=Path("eval/dataset/images"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/count_risk_analysis.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Measure review-flag coverage for exact and incorrect count scenes."""
    args = parse_args()
    manifest = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for filename, truth in manifest["images"].items():
        image = cv2.imread(str(args.images / filename))
        if image is None:
            raise ValueError(f"Cannot read evaluation image: {filename}")
        image_height, image_width = image.shape[:2]
        detections = run_inference(image)
        predicted = len(detections)
        expected = int(truth["expected_visible_count"])
        suspicious_pairs = []
        if predicted <= FRAGMENT_MAX_DETECTIONS:
            suspicious_pairs = find_suspicious_fragment_pairs(
                [detection["bbox_xyxy"] for detection in detections],
                image_width,
                image_height,
                max_gap_ratio=FRAGMENT_MAX_GAP_RATIO,
                min_axis_overlap=FRAGMENT_MIN_AXIS_OVERLAP,
                max_area_ratio=FRAGMENT_MAX_AREA_RATIO,
                max_iou=FRAGMENT_MAX_IOU,
            )
        rows.append(
            {
                "filename": filename,
                "expected_count": expected,
                "predicted_count": predicted,
                "count_error": predicted - expected,
                "suspicious_fragment_pairs": len(suspicious_pairs),
                "high_detection_count": predicted >= HIGH_DETECTION_COUNT,
                "review_flagged": bool(suspicious_pairs)
                or predicted >= HIGH_DETECTION_COUNT,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    count_errors = [row for row in rows if row["count_error"] != 0]
    flagged_errors = [row for row in count_errors if row["review_flagged"]]
    flagged_exact = [
        row for row in rows if row["count_error"] == 0 and row["review_flagged"]
    ]
    print(f"reviewed_images={len(rows)}")
    print(f"count_error_images={len(count_errors)}")
    print(f"flagged_count_error_images={len(flagged_errors)}")
    print(f"flagged_exact_count_images={len(flagged_exact)}")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
