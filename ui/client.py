"""Typed HTTP boundary for the Streamlit frontend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import requests

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class BackendError(Exception):
    """Base class for safe, user-facing backend failures."""


class BackendUnavailableError(BackendError):
    """Raised when the analysis service cannot be reached."""


class BackendTimeoutError(BackendError):
    """Raised when the analysis service exceeds the request timeout."""


class InvalidBackendResponseError(BackendError):
    """Raised when the service response does not match the expected contract."""


def normalize_backend_url(value: str) -> str:
    """Validate and normalize the configured backend base URL."""
    normalized = value.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("BACKEND_URL must begin with http:// or https://")
    return normalized


def validate_upload(content_type: str, size_bytes: int) -> str | None:
    """Return an operator-safe validation message, or ``None`` if valid."""
    if content_type not in SUPPORTED_IMAGE_TYPES:
        return "Please upload a JPEG or PNG image."
    if size_bytes == 0:
        return "The selected image is empty. Please choose another file."
    if size_bytes > MAX_UPLOAD_BYTES:
        return "The image is larger than 10 MB. Please choose a smaller file."
    return None


def analyze_image(
    backend_url: str,
    filename: str,
    content_type: str,
    image_bytes: bytes,
    *,
    connect_timeout_seconds: float = 65.0,
    read_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Submit one image and validate the minimum inference response contract."""
    url = f"{normalize_backend_url(backend_url)}/v1/box-intake/infer"
    try:
        response = requests.post(
            url,
            files={"file": (filename, image_bytes, content_type)},
            timeout=(connect_timeout_seconds, read_timeout_seconds),
        )
    except requests.Timeout as error:
        raise BackendTimeoutError(
            "Analysis took too long. The service may still be waking up."
        ) from error
    except requests.RequestException as error:
        raise BackendUnavailableError(
            "The analysis service is not responding. Try again shortly."
        ) from error

    if not response.ok:
        detail = _response_detail(response)
        raise BackendError(detail)

    try:
        payload = response.json()
    except requests.JSONDecodeError as error:
        raise InvalidBackendResponseError(
            "The analysis service returned an unreadable response."
        ) from error
    if not isinstance(payload, dict):
        raise InvalidBackendResponseError(
            "The analysis service returned an unexpected response."
        )
    _validate_result(payload)
    return payload


def _response_detail(response: requests.Response) -> str:
    """Extract a safe API error without exposing an internal traceback."""
    try:
        payload = response.json()
    except requests.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping) and isinstance(payload.get("detail"), str):
        return str(payload["detail"])
    if response.status_code >= 500:
        return "The analysis service is temporarily unavailable. Try again shortly."
    return "The image could not be analyzed. Check the file and try again."


def _validate_result(payload: Mapping[str, Any]) -> None:
    """Check fields the UI relies on before rendering them."""
    expected_types: dict[str, type[Any]] = {
        "visible_box_count": int,
        "size_summary": dict,
        "confidence_score": (int, float),  # type: ignore[dict-item]
        "human_review_required": bool,
        "review_reasons": list,
        "annotated_image_png_b64": str,
    }
    for field, expected_type in expected_types.items():
        if field not in payload or not isinstance(payload[field], expected_type):
            raise InvalidBackendResponseError(
                "The analysis service returned an incomplete result."
            )
