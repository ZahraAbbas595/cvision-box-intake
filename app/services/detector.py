from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from app.config import CONF_THRESHOLD, IOU_THRESHOLD

torch.set_num_threads(1)

_model: YOLO | None = None
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "carton_yolov8n_best.pt"


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
        _model.to("cpu")
    return _model


def run_inference(image_array: np.ndarray) -> list[dict[str, object]]:
    model = get_model()
    results = model(
        image_array,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device="cpu",
        verbose=False,
    )
    result = results[0]
    detections: list[dict[str, object]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "bbox_xyxy": [
                        round(x1),
                        round(y1),
                        round(x2),
                        round(y2),
                    ],
                    "confidence": round(float(box.conf[0]), 3),
                }
            )
    return detections
