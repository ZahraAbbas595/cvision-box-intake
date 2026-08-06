from ui.client import MAX_UPLOAD_BYTES, normalize_backend_url, validate_upload
from ui.presentation import explain_review_reasons


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
