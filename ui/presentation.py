"""Presentation helpers that translate API evidence for intake operators."""

from __future__ import annotations

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
    "count_mismatch_with_expected": (
        "The visible count does not match the expected count."
    ),
}


def explain_review_reasons(reason_codes: list[str]) -> list[str]:
    """Map stable API reason codes to concise, plain-language explanations."""
    return [
        REVIEW_REASON_MESSAGES.get(
            code, "The result contains an unfamiliar review signal."
        )
        for code in reason_codes
    ]
