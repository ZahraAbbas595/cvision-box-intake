from unittest.mock import patch

import requests

from ui.client import (
    MAX_UPLOAD_BYTES,
    BackendTimeoutError,
    BackendUnavailableError,
    analyze_image,
    normalize_backend_url,
    validate_upload,
)
from ui.presentation import explain_quality_flags, explain_review_reasons


def test_normalize_backend_url_removes_trailing_slash() -> None:
    assert normalize_backend_url("https://api.example.com/") == (
        "https://api.example.com"
    )


def test_normalize_backend_url_rejects_non_http_value() -> None:
    try:
        normalize_backend_url("api.example.com")
    except ValueError as error:
        assert "http:// or https://" in str(error)
    else:
        raise AssertionError("non-HTTP backend URL should be rejected")


def test_validate_upload_rejects_type_and_size_before_request() -> None:
    assert validate_upload("image/gif", 100) == "Please upload a JPEG or PNG image."
    assert validate_upload("image/jpeg", MAX_UPLOAD_BYTES + 1) is not None
    assert validate_upload("image/png", 100) is None


def test_review_reasons_are_presented_in_plain_language() -> None:
    messages = explain_review_reasons(
        ["possible_occlusion", "possible_fragmented_detections"]
    )

    assert messages == [
        "Some boxes overlap and may be hidden behind each other.",
        "One box may have been split into multiple detected regions.",
    ]


def test_quality_flags_ask_for_a_better_image_in_plain_language() -> None:
    messages = explain_quality_flags(
        ["possible_blur", "underexposed", "overexposed", "low_resolution"]
    )

    assert messages == [
        "the image may be blurry",
        "the image is too dark",
        "the image is too bright",
        "the image resolution is too low",
    ]


def test_timeout_message_is_safe_and_actionable() -> None:
    with patch("ui.client.requests.post", side_effect=requests.Timeout):
        try:
            analyze_image("https://api.example.com", "box.jpg", "image/jpeg", b"x")
        except BackendTimeoutError as error:
            assert str(error) == (
                "Analysis took too long. The service may still be waking up."
            )
        else:
            raise AssertionError("timeout should raise BackendTimeoutError")


def test_unavailable_backend_message_is_safe_and_actionable() -> None:
    with patch("ui.client.requests.post", side_effect=requests.ConnectionError):
        try:
            analyze_image("https://api.example.com", "box.jpg", "image/jpeg", b"x")
        except BackendUnavailableError as error:
            assert str(error) == (
                "The analysis service is not responding. Try again shortly."
            )
        else:
            raise AssertionError(
                "connection failure should raise BackendUnavailableError"
            )
