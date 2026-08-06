"""Operator-facing Streamlit application for visual carton intake."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import streamlit as st

from ui.client import BackendError, analyze_image, validate_upload
from ui.presentation import explain_review_reasons

st.set_page_config(page_title="CVision Box Intake", page_icon="📦", layout="wide")


def configured_backend_url() -> str:
    """Read the backend URL from Streamlit secrets or the process environment."""
    secret_url = st.secrets.get("BACKEND_URL", "")
    return str(secret_url or os.getenv("BACKEND_URL", "")).strip()


def render_result(result: dict[str, Any]) -> None:
    """Render evidence and review guidance from a validated API result."""
    count_col, confidence_col, review_col = st.columns(3)
    count_col.metric("Visible boxes", int(result["visible_box_count"]))
    confidence_col.metric("Operational confidence", f"{result['confidence_score']:.0%}")
    review_required = bool(result["human_review_required"])
    review_col.metric("Human review", "Required" if review_required else "Not required")

    if review_required:
        st.warning("Please review this result before using the visible count.")
        codes = [str(code) for code in result["review_reasons"]]
        for explanation in explain_review_reasons(codes):
            st.markdown(f"- {explanation}")
    else:
        st.success("No automatic review signals were triggered.")

    annotated_bytes = base64.b64decode(result["annotated_image_png_b64"], validate=True)
    st.subheader("Annotated evidence")
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

    uploaded = st.file_uploader("Upload a carton image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Choose a JPEG or PNG image up to 10 MB to begin.")
        return

    image_bytes = uploaded.getvalue()
    content_type = uploaded.type or "application/octet-stream"
    validation_error = validate_upload(content_type, len(image_bytes))
    if validation_error:
        st.error(validation_error)
        return

    st.subheader("Original image")
    st.image(image_bytes, use_container_width=True)
    if not st.button("Analyze image", type="primary"):
        return

    try:
        with st.spinner("Waking up the service if needed and analyzing the image…"):
            result = analyze_image(
                backend_url, uploaded.name, content_type, image_bytes
            )
        render_result(result)
    except BackendError as error:
        st.error(str(error))
        st.info("Use the Analyze image button to retry.")
    except (ValueError, TypeError, base64.binascii.Error):
        st.error("The analysis result could not be displayed. Please try again.")


if __name__ == "__main__":
    main()
