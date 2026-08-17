"""Tests for the optional, fail-safe visual review service."""

import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

import numpy as np

from app.services.visual_review import assess_visual_review


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _response_with_assessment(assessment: dict) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": json.dumps(assessment)}]},
            }
        ]
    }


def test_visual_review_returns_unavailable_without_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [],
            ["no_boxes_detected"],
            [],
            model="test-model",
            timeout_seconds=1,
        )

    assert result["status"] == "unavailable"
    assert result["visual_risk_reasons"] == []
    assert result["novel_reason"] is None


def test_visual_review_accepts_allow_listed_structured_output() -> None:
    assessment = {
        "visual_review_required": True,
        "visual_risk_reasons": ["possible_occlusion", "reflection_or_shadow"],
        "summary": "Several regions overlap and one has strong glare.",
        "operator_guidance": "Check the center stack against the original scene.",
        "novel_reason": "",
        "confidence": 0.81,
    }
    api_response = _FakeResponse(_response_with_assessment(assessment))

    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        patch("urllib.request.urlopen", return_value=api_response) as urlopen,
    ):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [{"id": 1, "confidence": 0.9, "touches_edge": False}],
            [],
            [],
            model="test-model",
            timeout_seconds=3,
        )

    assert result["status"] == "completed"
    assert result["visual_review_required"] is True
    assert result["visual_risk_reasons"] == [
        "possible_occlusion",
        "reflection_or_shadow",
    ]
    request = urlopen.call_args.args[0]
    sent_payload = json.loads(request.data.decode("utf-8"))
    assert "inline_data" in sent_payload["contents"][0]["parts"][0]
    prompt = sent_payload["contents"][0]["parts"][1]["text"]
    assert "mark confirmed non-carton" in prompt
    assert "Never change the API detections" in prompt
    assert "Do not call an entire detection a false positive" in prompt
    assert "Use unusual_non_carton_objects for windows" in prompt
    assert "Use other_visual_risk" in prompt
    assert "known code adequately describes" in prompt
    generation_config = sent_payload["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert (
        "other_visual_risk"
        in generation_config["responseSchema"]["properties"]["visual_risk_reasons"][
            "items"
        ]["enum"]
    )
    assert request.headers["X-goog-api-key"] == "test-key"
    assert request.full_url.endswith("/test-model:generateContent")


def test_visual_review_accepts_specific_novel_reason() -> None:
    assessment = {
        "visual_review_required": True,
        "visual_risk_reasons": ["other_visual_risk"],
        "summary": "A hanging strap crosses several carton boundaries.",
        "operator_guidance": "Move the strap and capture another image.",
        "novel_reason": (
            "A wide packing strap hangs in front of the stack and visually connects "
            "separate carton faces."
        ),
        "confidence": 0.76,
    }
    api_response = _FakeResponse(_response_with_assessment(assessment))

    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        patch("urllib.request.urlopen", return_value=api_response),
    ):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [],
            [],
            [],
            model="test-model",
            timeout_seconds=1,
        )

    assert result["status"] == "completed"
    assert result["visual_review_required"] is True
    assert result["visual_risk_reasons"] == ["other_visual_risk"]
    assert result["novel_reason"].startswith("A wide packing strap")


def test_visual_review_rejects_inconsistent_output() -> None:
    assessment = {
        "visual_review_required": False,
        "visual_risk_reasons": ["possible_occlusion"],
        "summary": "Possible overlap.",
        "operator_guidance": "Inspect the center stack.",
        "novel_reason": "",
        "confidence": 0.7,
    }
    api_response = _FakeResponse(_response_with_assessment(assessment))

    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        patch("urllib.request.urlopen", return_value=api_response),
    ):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [],
            [],
            [],
            model="test-model",
            timeout_seconds=1,
        )

    assert result["status"] == "unavailable"
    assert result["visual_review_required"] is False


def test_visual_review_rejects_other_risk_without_novel_reason() -> None:
    assessment = {
        "visual_review_required": True,
        "visual_risk_reasons": ["other_visual_risk"],
        "summary": "An uncategorized issue is present.",
        "operator_guidance": "Inspect the image manually.",
        "novel_reason": "",
        "confidence": 0.7,
    }
    api_response = _FakeResponse(_response_with_assessment(assessment))

    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        patch("urllib.request.urlopen", return_value=api_response),
    ):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [],
            [],
            [],
            model="test-model",
            timeout_seconds=1,
        )

    assert result["status"] == "unavailable"


def test_visual_review_rejects_unexpected_output_field() -> None:
    assessment = {
        "visual_review_required": False,
        "visual_risk_reasons": [],
        "summary": "No additional visual risk is evident.",
        "operator_guidance": "Verify the annotated regions before intake.",
        "novel_reason": "",
        "confidence": 0.8,
        "unexpected": "not part of the contract",
    }
    api_response = _FakeResponse(_response_with_assessment(assessment))

    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        patch("urllib.request.urlopen", return_value=api_response),
    ):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [],
            [],
            [],
            model="test-model",
            timeout_seconds=1,
        )

    assert result["status"] == "unavailable"


def test_visual_review_logs_safe_gemini_http_error() -> None:
    provider_error = urllib.error.HTTPError(
        "https://generativelanguage.googleapis.com/redacted",
        400,
        "Bad Request",
        {},
        BytesIO(
            json.dumps(
                {
                    "error": {
                        "status": "INVALID_ARGUMENT",
                        "message": "Unsupported schema field.",
                    }
                }
            ).encode("utf-8")
        ),
    )

    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "secret-key"}, clear=True),
        patch("urllib.request.urlopen", side_effect=provider_error),
        patch("app.services.visual_review.logger.warning") as warning,
    ):
        result = assess_visual_review(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [],
            [],
            [],
            model="test-model",
            timeout_seconds=1,
        )

    assert result["status"] == "unavailable"
    log_payload = json.loads(warning.call_args.args[1])
    assert log_payload == {
        "event": "visual_review_provider_error",
        "provider": "google_gemini",
        "model": "test-model",
        "http_status": 400,
        "api_status": "INVALID_ARGUMENT",
        "message": "Unsupported schema field.",
    }
    assert "secret-key" not in warning.call_args.args[1]
