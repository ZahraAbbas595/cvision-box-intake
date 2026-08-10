"""Presentation helpers that translate API evidence for intake operators."""

from __future__ import annotations

from typing import Any

REVIEW_REASON_MESSAGES = {
    "no_boxes_detected": "No clearly identifiable boxes were found.",
    "low_confidence_detection": (
        "At least one box detection is uncertain and should be checked."
    ),
    "boxes_cut_off_at_edge": (
        "Some boxes are cut off by the image edge and may not be fully visible."
    ),
    "high_detection_count": (
        "This is a crowded scene, so the visible count should be checked."
    ),
    "possible_fragmented_detections": (
        "One box may have been split into multiple detected regions."
    ),
    "possible_occlusion": ("Some boxes overlap and may be hidden behind each other."),
    "poor_image_quality": ("Image quality may make the visible count less reliable."),
    "reflection_or_shadow": (
        "A reflection or shadow may resemble a box or hide its visible boundary."
    ),
    "unusual_non_carton_objects": (
        "Other objects in the scene may be mistaken for cartons."
    ),
    "confusing_background": (
        "The background may make some carton boundaries difficult to distinguish."
    ),
    "damaged_or_deformed_cartons": (
        "Damaged or deformed cartons may not produce reliable detection regions."
    ),
    "ambiguous_small_objects": (
        "Some small visible regions are too ambiguous to count confidently."
    ),
    "other_visual_risk": (
        "The visual reviewer found an additional scene-specific counting risk."
    ),
    "count_mismatch_with_expected": (
        "The visible count does not match the expected count."
    ),
}

QUALITY_FLAG_MESSAGES = {
    "low_resolution": "the image resolution is too low",
    "possible_blur": "the image may be blurry",
    "underexposed": "the image is too dark",
    "overexposed": "the image is too bright",
}


def preferred_backend_url(environment_url: str, secret_url: str = "") -> str:
    """Prefer an explicit process URL over an optional Streamlit secret."""
    return environment_url.strip() or secret_url.strip()


def explain_review_reasons(reason_codes: list[str]) -> list[str]:
    """Map stable API reason codes to concise, plain-language explanations."""
    return [
        REVIEW_REASON_MESSAGES.get(
            code, "The result contains an unfamiliar review signal."
        )
        for code in reason_codes
    ]


def explain_quality_flags(flag_codes: list[str]) -> list[str]:
    """Translate image-quality codes into specific operator guidance."""
    return [
        QUALITY_FLAG_MESSAGES.get(code, "the image has an unknown quality issue")
        for code in flag_codes
    ]


def review_display_state(
    reason_codes: list[str],
    quality_flags: list[str],
    assessment: Any,
) -> tuple[bool, list[str]]:
    """Reconcile deterministic and model review signals for operator display."""
    visual_review_required = (
        isinstance(assessment, dict)
        and assessment.get("status") == "completed"
        and assessment.get("visual_review_required") is True
    )
    displayed_codes = [
        code
        for code in reason_codes
        if not (code == "poor_image_quality" and quality_flags)
    ]
    return visual_review_required, displayed_codes
