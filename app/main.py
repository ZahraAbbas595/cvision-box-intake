"""FastAPI application and carton-inference HTTP endpoints."""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from app.config import (
    BLUR_VARIANCE_THRESHOLD,
    BRIGHT_MEAN_THRESHOLD,
    CONF_THRESHOLD,
    DARK_MEAN_THRESHOLD,
    EDGE_MARGIN_PX,
    EDGE_REVIEW_RATIO,
    FRAGMENT_MAX_AREA_RATIO,
    FRAGMENT_MAX_DETECTIONS,
    FRAGMENT_MAX_GAP_RATIO,
    FRAGMENT_MAX_IOU,
    FRAGMENT_MIN_AXIS_OVERLAP,
    FRAGMENT_MIN_PAIRS,
    IOU_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_UPLOAD_BYTES,
    MIN_IMAGE_DIMENSION,
    MODEL_BACKEND,
    MODEL_NAME,
    MODEL_VERSION,
    SCHEMA_VERSION,
    SERVICE_VERSION,
    VISUAL_REVIEW_ENABLED,
    VISUAL_REVIEW_MODEL,
    VISUAL_REVIEW_TIMEOUT_SECONDS,
)
from app.schemas import HealthResponse, VersionResponse
from app.services.count_risk import find_suspicious_fragment_pairs
from app.services.detector import run_inference
from app.services.quality import assess_image_quality
from app.services.runtime import memory_snapshot
from app.services.visual_review import assess_visual_review

app = FastAPI(title="CVision Box Intake API")
logger = logging.getLogger("uvicorn.error")


@app.get("/")
def root() -> dict[str, str]:
    """Describe the service when its base URL is opened in a browser."""
    return {
        "service": "CVision Box Intake API",
        "status": "ok",
        "health": "/health",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a lightweight service liveness response."""
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """Return the deployed service, schema, model, and backend versions."""
    return VersionResponse(
        service_version=SERVICE_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_backend=MODEL_BACKEND,
        schema_version=SCHEMA_VERSION,
    )


@app.post("/v1/box-intake/infer")
async def infer(response: Response, file: UploadFile = File(...)) -> dict:
    """Validate an uploaded image and return carton-intake evidence."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Send JPEG or PNG."
        )
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Image too large. Maximum size is 10MB."
        )
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=422, detail="Cannot read image file. It may be corrupted."
        )
    img_h, img_w = image.shape[:2]
    quality_flags = assess_image_quality(
        image,
        min_dimension=MIN_IMAGE_DIMENSION,
        blur_variance_threshold=BLUR_VARIANCE_THRESHOLD,
        dark_mean_threshold=DARK_MEAN_THRESHOLD,
        bright_mean_threshold=BRIGHT_MEAN_THRESHOLD,
    )
    try:
        detections = run_inference(image)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.exception(
            "%s",
            json.dumps(
                {
                    "event": "inference_failed",
                    "request_id": request_id,
                    "model_backend": MODEL_BACKEND,
                    "error": str(error),
                }
            ),
        )
        raise HTTPException(
            status_code=503,
            detail="Model inference is temporarily unavailable.",
        ) from error

    detection_list = []
    size_summary = {"small": 0, "medium": 0, "large": 0}
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        size_class = (
            "small"
            if area_ratio < 0.03
            else ("large" if area_ratio > 0.10 else "medium")
        )
        size_summary[size_class] += 1
        touches_edge = (
            x1 <= EDGE_MARGIN_PX
            or y1 <= EDGE_MARGIN_PX
            or x2 >= img_w - EDGE_MARGIN_PX
            or y2 >= img_h - EDGE_MARGIN_PX
        )
        detection_list.append(
            {
                "id": i + 1,
                "bbox_xyxy": detection["bbox_xyxy"],
                "confidence": detection["confidence"],
                "size_class": size_class,
                "area_ratio": round(area_ratio, 4),
                "touches_edge": touches_edge,
            }
        )

    review_reasons = []
    if quality_flags:
        review_reasons.append("poor_image_quality")
    if len(detection_list) == 0:
        review_reasons.append("no_boxes_detected")
    if any(
        detection["confidence"] < LOW_CONFIDENCE_THRESHOLD
        for detection in detection_list
    ):
        review_reasons.append("low_confidence_detection")
    edge_count = sum(1 for detection in detection_list if detection["touches_edge"])
    if len(detection_list) > 0 and edge_count / len(detection_list) > EDGE_REVIEW_RATIO:
        review_reasons.append("boxes_cut_off_at_edge")
    suspicious_pairs = []
    if len(detection_list) <= FRAGMENT_MAX_DETECTIONS:
        suspicious_pairs = find_suspicious_fragment_pairs(
            [detection["bbox_xyxy"] for detection in detection_list],
            img_w,
            img_h,
            max_gap_ratio=FRAGMENT_MAX_GAP_RATIO,
            min_axis_overlap=FRAGMENT_MIN_AXIS_OVERLAP,
            max_area_ratio=FRAGMENT_MAX_AREA_RATIO,
            max_iou=FRAGMENT_MAX_IOU,
        )
    if len(suspicious_pairs) >= FRAGMENT_MIN_PAIRS:
        review_reasons.append("possible_fragmented_detections")

    if len(detection_list) == 0:
        confidence_score = 0.0
    else:
        confidence_score = round(
            sum(detection["confidence"] for detection in detection_list)
            / len(detection_list),
            2,
        )

    annotated = image.copy()
    for detection in detection_list:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{detection['size_class']} {detection['confidence']:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    if max(img_h, img_w) > 1024:
        scale = 1024 / max(img_h, img_w)
        annotated = cv2.resize(annotated, (int(img_w * scale), int(img_h * scale)))
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    review_assessment = {
        "status": "disabled",
        "provider": "google_gemini",
        "model": None,
        "visual_review_required": False,
        "visual_risk_reasons": [],
        "summary": None,
        "operator_guidance": None,
        "novel_reason": None,
        "confidence": None,
    }
    if VISUAL_REVIEW_ENABLED:
        review_assessment = await asyncio.to_thread(
            assess_visual_review,
            annotated,
            detection_list,
            review_reasons,
            quality_flags,
            model=VISUAL_REVIEW_MODEL,
            timeout_seconds=VISUAL_REVIEW_TIMEOUT_SECONDS,
        )
        for reason in review_assessment["visual_risk_reasons"]:
            if reason not in review_reasons:
                review_reasons.append(reason)
        if review_assessment["novel_reason"]:
            logger.warning(
                "%s",
                json.dumps(
                    {
                        "event": "novel_visual_risk_detected",
                        "request_id": request_id,
                        "reason_code": "other_visual_risk",
                        "novel_reason": review_assessment["novel_reason"],
                        "model": review_assessment["model"],
                        "provider": review_assessment["provider"],
                        "confidence": review_assessment["confidence"],
                    }
                ),
            )

    processing_time_ms = round((time.time() - start_time) * 1000)
    runtime_memory = memory_snapshot()
    if runtime_memory["current_rss_mb"] is not None:
        response.headers["X-Process-RSS-MB"] = str(runtime_memory["current_rss_mb"])
    response.headers["Server-Timing"] = f"inference;dur={processing_time_ms}"
    logger.info(
        "%s",
        json.dumps(
            {
                "event": "inference_completed",
                "request_id": request_id,
                "model_backend": MODEL_BACKEND,
                "visible_box_count": len(detection_list),
                "review_reasons": review_reasons,
                "processing_time_ms": processing_time_ms,
                **runtime_memory,
            }
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": "box_intake_scan",
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": {
            "width": img_w,
            "height": img_h,
            "quality_flags": quality_flags,
        },
        "detections": detection_list,
        "visible_box_count": len(detection_list),
        "size_summary": size_summary,
        "confidence_score": confidence_score,
        "human_review_required": len(review_reasons) > 0,
        "review_reasons": review_reasons,
        "review_assessment": review_assessment,
        "count_risk": {
            "suspicious_fragment_pair_count": len(suspicious_pairs),
        },
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "backend": MODEL_BACKEND,
        },
        "service": {"version": SERVICE_VERSION},
        "runtime_memory": runtime_memory,
        "processing_time_ms": processing_time_ms,
        "annotated_image_png_b64": annotated_b64,
    }
