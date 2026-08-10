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
from ui.presentation import (
    explain_quality_flags,
    explain_review_reasons,
    preferred_backend_url,
    review_display_state,
)


def test_normalize_backend_url_removes_trailing_slash() -> None:
    assert normalize_backend_url("https://api.example.com/") == (
        "https://api.example.com"
    )


def test_backend_configuration_prefers_environment_over_secret() -> None:
    assert preferred_backend_url(
        "http://127.0.0.1:8000", "https://secret.example.com"
    ) == ("http://127.0.0.1:8000")


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


def test_visual_review_reasons_are_presented_in_plain_language() -> None:
    messages = explain_review_reasons(
        [
            "reflection_or_shadow",
            "unusual_non_carton_objects",
            "other_visual_risk",
        ]
    )

    assert messages == [
        "A reflection or shadow may resemble a box or hide its visible boundary.",
        "Other objects in the scene may be mistaken for cartons.",
        "The visual reviewer found an additional scene-specific counting risk.",
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


def test_quality_only_review_does_not_show_no_risk_model_guidance() -> None:
    assessment = {
        "status": "completed",
        "visual_review_required": False,
        "summary": "The carton has no identifiable visual risks.",
        "operator_guidance": "Process the carton as usual.",
    }

    visual_review_required, displayed_codes = review_display_state(
        ["poor_image_quality"], ["possible_blur"], assessment
    )

    assert visual_review_required is False
    assert displayed_codes == []


def test_visual_risk_remains_visible_alongside_quality_banner() -> None:
    assessment = {"status": "completed", "visual_review_required": True}

    visual_review_required, displayed_codes = review_display_state(
        ["poor_image_quality", "reflection_or_shadow"],
        ["underexposed"],
        assessment,
    )

    assert visual_review_required is True
    assert displayed_codes == ["reflection_or_shadow"]


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
