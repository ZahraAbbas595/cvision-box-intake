"""Central configuration values loaded from environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse an opt-in boolean environment setting."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_path(value: str) -> Path:
    """Resolve an absolute or repository-relative model path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.30"))
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "pytorch").lower()
MODEL_PATH = _resolve_model_path(
    os.getenv("MODEL_PATH", "models/carton_yolov8n_truck_best.pt")
)
INFERENCE_IMAGE_SIZE = int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "200"))
BLUR_VARIANCE_THRESHOLD = float(os.getenv("BLUR_VARIANCE_THRESHOLD", "50.0"))
DARK_MEAN_THRESHOLD = float(os.getenv("DARK_MEAN_THRESHOLD", "40.0"))
BRIGHT_MEAN_THRESHOLD = float(os.getenv("BRIGHT_MEAN_THRESHOLD", "215.0"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45"))
EDGE_MARGIN_PX = int(os.getenv("EDGE_MARGIN_PX", "5"))
EDGE_REVIEW_RATIO = float(os.getenv("EDGE_REVIEW_RATIO", "0.20"))
FRAGMENT_MAX_GAP_RATIO = float(os.getenv("FRAGMENT_MAX_GAP_RATIO", "0.04"))
FRAGMENT_MIN_AXIS_OVERLAP = float(os.getenv("FRAGMENT_MIN_AXIS_OVERLAP", "0.60"))
FRAGMENT_MAX_AREA_RATIO = float(os.getenv("FRAGMENT_MAX_AREA_RATIO", "4.0"))
FRAGMENT_MAX_IOU = float(os.getenv("FRAGMENT_MAX_IOU", "0.02"))
FRAGMENT_MAX_DETECTIONS = int(os.getenv("FRAGMENT_MAX_DETECTIONS", "2"))
FRAGMENT_MIN_PAIRS = int(os.getenv("FRAGMENT_MIN_PAIRS", "1"))
VISUAL_REVIEW_ENABLED = _env_flag("VISUAL_REVIEW_ENABLED")
VISUAL_REVIEW_MODEL = os.getenv("VISUAL_REVIEW_MODEL", "gemini-3.1-flash-lite")
VISUAL_REVIEW_TIMEOUT_SECONDS = float(
    os.getenv("VISUAL_REVIEW_TIMEOUT_SECONDS", "12.0")
)
SERVICE_VERSION = "0.9.0"
MODEL_NAME = os.getenv("MODEL_NAME", "carton-yolov8n-truck")
MODEL_VERSION = os.getenv("MODEL_VERSION", "ft-truck-v1")
SCHEMA_VERSION = "1.1"
