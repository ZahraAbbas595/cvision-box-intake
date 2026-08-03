import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.services.detector import ONNX_MODEL_PATH, _nms, _prepare_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ONNX count thresholds.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("eval/dataset/ground_truth.json"),
    )
    parser.add_argument("--images", type=Path, default=Path("eval/dataset/images"))
    parser.add_argument("--model", type=Path, default=ONNX_MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/onnx_threshold_sweep.csv"),
    )
    return parser.parse_args()


def decode_candidates(
    raw_output: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.squeeze(raw_output, axis=0)
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    scores = prediction[:, 4:].max(axis=1)
    centers = prediction[:, :2]
    sizes = prediction[:, 2:4]
    boxes = np.column_stack(
        (
            centers[:, 0] - sizes[:, 0] / 2,
            centers[:, 1] - sizes[:, 1] / 2,
            centers[:, 0] + sizes[:, 0] / 2,
            centers[:, 1] + sizes[:, 1] / 2,
        )
    )
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    image_height, image_width = image.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_height)
    return boxes, scores


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    predictions: list[tuple[int, np.ndarray, np.ndarray]] = []
    for filename, truth in manifest["images"].items():
        image = cv2.imread(str(args.images / filename))
        if image is None:
            raise ValueError(f"Cannot read evaluation image: {filename}")
        tensor, scale, pad_x, pad_y = _prepare_input(image)
        raw_output = session.run(None, {input_name: tensor})[0]
        boxes, scores = decode_candidates(
            raw_output,
            scale,
            pad_x,
            pad_y,
            image,
        )
        predictions.append((int(truth["expected_visible_count"]), boxes, scores))

    rows: list[dict[str, float | int]] = []
    for confidence_step in range(35, 56):
        confidence = confidence_step / 100
        for iou_step in range(20, 51, 5):
            iou = iou_step / 100
            errors: list[int] = []
            for expected, boxes, scores in predictions:
                accepted = scores >= confidence
                predicted = len(_nms(boxes[accepted], scores[accepted], iou))
                errors.append(predicted - expected)
            exact = sum(error == 0 for error in errors)
            rows.append(
                {
                    "confidence": confidence,
                    "iou": iou,
                    "exact_count_accuracy": round(exact / len(errors), 4),
                    "mean_absolute_count_error": round(
                        statistics.fmean(abs(error) for error in errors), 4
                    ),
                    "overcount_images": sum(error > 0 for error in errors),
                    "undercount_images": sum(error < 0 for error in errors),
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
    for row in rows[:10]:
        print(row)
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
