"""Operator-facing Streamlit application for visual carton intake."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from ui.client import BackendError, analyze_image, validate_upload
    from ui.presentation import explain_review_reasons, review_display_state
except ModuleNotFoundError:  # Streamlit Cloud can execute this file as a script.
    from client import BackendError, analyze_image, validate_upload
    from presentation import explain_review_reasons, review_display_state

QUALITY_FLAG_MESSAGES = {
    "low_resolution": "the image resolution is too low",
    "possible_blur": "the image may be blurry",
    "underexposed": "the image is too dark",
    "overexposed": "the image is too bright",
}

st.set_page_config(page_title="CVision Box Intake", page_icon="📦", layout="wide")


def configured_backend_url() -> str:
    """Read the backend URL from Streamlit secrets or the process environment."""
    environment_url = os.getenv("BACKEND_URL", "").strip()
    if environment_url:
        return environment_url
    try:
        return str(st.secrets.get("BACKEND_URL", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def render_result(result: dict[str, Any], original_bytes: bytes) -> None:
    """Render evidence and review guidance from a validated API result."""
    image_metadata = result.get("image", {})
    quality_flags = (
        [str(flag) for flag in image_metadata.get("quality_flags", [])]
        if isinstance(image_metadata, dict)
        else []
    )
    if quality_flags:
        quality_details = "; ".join(
            QUALITY_FLAG_MESSAGES.get(flag, "the image has an unknown quality issue")
            for flag in quality_flags
        )
        st.error(
            "Image quality is not good enough for a reliable result: "
            f"{quality_details}. Please retake the photo or upload a clearer image, "
            "then analyze it again."
        )

    count_col, confidence_col, review_col = st.columns(3)
    count_col.metric("Visible boxes", int(result["visible_box_count"]))
    confidence_col.metric("Operational confidence", f"{result['confidence_score']:.0%}")
    review_required = bool(result["human_review_required"])
    review_col.metric("Human review", "Required" if review_required else "Not required")

    if review_required:
        codes = [str(code) for code in result["review_reasons"]]
        assessment = result.get("review_assessment")
        visual_review_required, displayed_codes = review_display_state(
            codes, quality_flags, assessment
        )

        if visual_review_required:
            summary = assessment.get("summary")
            if isinstance(summary, str) and summary:
                st.markdown(f"**Why review is needed:** {summary}")
            guidance = assessment.get("operator_guidance")
            if isinstance(guidance, str) and guidance:
                st.info(f"**Suggested check:** {guidance}")
            if displayed_codes or assessment.get("novel_reason"):
                with st.expander("Technical review signals"):
                    for explanation in explain_review_reasons(displayed_codes):
                        st.markdown(f"- {explanation}")
                    novel_reason = assessment.get("novel_reason")
                    if isinstance(novel_reason, str) and novel_reason:
                        st.markdown(f"- Additional observation: {novel_reason}")
        elif displayed_codes:
            st.markdown("**Why review is needed:**")
            for explanation in explain_review_reasons(displayed_codes):
                st.markdown(f"- {explanation}")
    else:
        st.success("No automatic review signals were triggered.")

    annotated_bytes = base64.b64decode(result["annotated_image_png_b64"], validate=True)
    st.subheader("Visual comparison")
    original_col, annotated_col = st.columns(2, gap="medium")
    with original_col:
        st.caption("Original")
        st.image(original_bytes, use_container_width=True)
    with annotated_col:
        st.caption("Annotated evidence")
        st.image(annotated_bytes, use_container_width=True)

    summary = result["size_summary"]
    st.subheader("Image-relative size summary")
    size_cols = st.columns(3)
    for column, size_name in zip(size_cols, ("small", "medium", "large"), strict=True):
        column.metric(size_name.title(), int(summary.get(size_name, 0)))
    st.caption(
        "Sizes are relative to the image, not physical measurements. Camera distance "
        "and angle can change the assigned class."
    )

    result_json = json.dumps(result, indent=2)
    with st.expander("Structured result"):
        st.json(result)
    st.download_button(
        "Download JSON evidence",
        result_json,
        file_name=f"box-intake-{result.get('request_id', 'result')}.json",
        mime="application/json",
    )


def main() -> None:
    """Render the upload, analysis, and evidence workflow."""
    st.title("CVision Box Intake")
    st.caption(
        "Research prototype for estimating visible carton counts before barcode "
        "or OCR confirmation. Results are evidence for review, not production "
        "decisions."
    )
    backend_url = configured_backend_url()
    if not backend_url:
        st.error("The analysis service is not configured. Set BACKEND_URL and restart.")
        st.stop()

    source = st.radio(
        "Choose an image source",
        ("Upload a file", "Take a photo"),
        horizontal=True,
    )
    if source == "Take a photo":
        uploaded = st.camera_input("Take a carton photo")
    else:
        uploaded = st.file_uploader(
            "Upload a carton image", type=["jpg", "jpeg", "png"]
        )
    if uploaded is None:
        if source == "Take a photo":
            st.info("Take a clear, well-lit photo of the cartons to begin.")
        else:
            st.info("Choose a JPEG or PNG image up to 10 MB to begin.")
        return

    image_bytes = uploaded.getvalue()
    content_type = uploaded.type or "application/octet-stream"
    validation_error = validate_upload(content_type, len(image_bytes))
    if validation_error:
        st.error(validation_error)
        return

    st.subheader("Original image preview")
    st.image(image_bytes, width=420)
    if not st.button("Analyze image", type="primary"):
        return

    try:
        with st.spinner("Waking up the service if needed and analyzing the image…"):
            result = analyze_image(
                backend_url, uploaded.name, content_type, image_bytes
            )
        render_result(result, image_bytes)
    except BackendError as error:
        st.error(str(error))
        st.info("Use the Analyze image button to retry.")
    except (ValueError, TypeError, base64.binascii.Error):
        st.error("The analysis result could not be displayed. Please try again.")


if __name__ == "__main__":
    main()
