"""PyTorch and ONNX carton-detection backends and post-processing."""

from typing import Protocol

import cv2
import numpy as np

from app.config import (
    CONF_THRESHOLD,
    INFERENCE_IMAGE_SIZE,
    IOU_THRESHOLD,
    MODEL_BACKEND,
    MODEL_PATH,
)

ONNX_MODEL_PATH = MODEL_PATH.with_suffix(".onnx")
MODEL_STRIDE = 32


class Detector(Protocol):
    """Backend-independent carton detector contract."""

    def predict(self, image_array: np.ndarray) -> list[dict[str, object]]:
        """Return accepted carton detections for a decoded BGR image."""
        ...


class PyTorchDetector:
    """Ultralytics/PyTorch detector used for local development."""

    def __init__(self) -> None:
        import torch

        from ultralytics import YOLO

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"PyTorch model not found at {MODEL_PATH}.")
        torch.set_num_threads(1)
        self._model = YOLO(MODEL_PATH)
        self._model.to("cpu")

    def predict(self, image_array: np.ndarray) -> list[dict[str, object]]:
        """Run Ultralytics inference and normalize its detection response."""
        results = self._model(
            image_array,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=INFERENCE_IMAGE_SIZE,
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
                        "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
                        "confidence": round(float(box.conf[0]), 3),
                    }
                )
        return detections


class OnnxDetector:
    """Lightweight ONNX Runtime detector used by production."""

    def __init__(self) -> None:
        import onnxruntime as ort

        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                "ONNX model not found at "
                f"{ONNX_MODEL_PATH}. Run scripts/export_onnx.py."
            )
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, image_array: np.ndarray) -> list[dict[str, object]]:
        """Run ONNX inference with production preprocessing and NMS."""
        tensor, scale, pad_x, pad_y = _prepare_input(image_array)
        raw_output = self._session.run(None, {self._input_name: tensor})[0]
        boxes, scores = _decode_output(raw_output, scale, pad_x, pad_y, image_array)
        kept = _nms(boxes, scores, IOU_THRESHOLD)
        return [
            {
                "bbox_xyxy": [round(value) for value in boxes[index]],
                "confidence": round(float(scores[index]), 3),
            }
            for index in kept
        ]


def _prepare_input(
    image_array: np.ndarray,
) -> tuple[np.ndarray, float, int, int]:
    image_height, image_width = image_array.shape[:2]
    scale = min(
        INFERENCE_IMAGE_SIZE / image_width,
        INFERENCE_IMAGE_SIZE / image_height,
    )
    resized_width = round(image_width * scale)
    resized_height = round(image_height * scale)
    resized = cv2.resize(image_array, (resized_width, resized_height))
    horizontal_padding = (INFERENCE_IMAGE_SIZE - resized_width) % MODEL_STRIDE
    vertical_padding = (INFERENCE_IMAGE_SIZE - resized_height) % MODEL_STRIDE
    pad_x = round(horizontal_padding / 2 - 0.1)
    pad_right = round(horizontal_padding / 2 + 0.1)
    pad_y = round(vertical_padding / 2 - 0.1)
    pad_bottom = round(vertical_padding / 2 + 0.1)
    canvas = cv2.copyMakeBorder(
        resized,
        pad_y,
        pad_bottom,
        pad_x,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), scale, pad_x, pad_y


def _decode_output(
    raw_output: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    image_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.squeeze(raw_output, axis=0)
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    scores = prediction[:, 4:].max(axis=1)
    accepted = scores >= CONF_THRESHOLD
    prediction = prediction[accepted]
    scores = scores[accepted]
    if len(prediction) == 0:
        return np.empty((0, 4), dtype=np.float32), scores
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
    image_height, image_width = image_array.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_height)
    return boxes, scores


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while len(order) > 0:
        current = int(order[0])
        kept.append(current)
        intersection_x1 = np.maximum(x1[current], x1[order[1:]])
        intersection_y1 = np.maximum(y1[current], y1[order[1:]])
        intersection_x2 = np.minimum(x2[current], x2[order[1:]])
        intersection_y2 = np.minimum(y2[current], y2[order[1:]])
        intersection = np.maximum(intersection_x2 - intersection_x1, 0) * np.maximum(
            intersection_y2 - intersection_y1, 0
        )
        union = areas[current] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return kept


_detector: Detector | None = None


def get_detector() -> Detector:
    """Lazily construct and cache the configured inference backend."""
    global _detector
    if _detector is None:
        if MODEL_BACKEND == "pytorch":
            _detector = PyTorchDetector()
        elif MODEL_BACKEND == "onnx":
            _detector = OnnxDetector()
        else:
            raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")
    return _detector


def run_inference(image_array: np.ndarray) -> list[dict[str, object]]:
    """Run the configured detector for a decoded BGR image."""
    return get_detector().predict(image_array)
