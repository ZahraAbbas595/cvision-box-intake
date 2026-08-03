import base64
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MODEL_NAME,
    MODEL_VERSION,
    SCHEMA_VERSION,
    SERVICE_VERSION,
)
from app.schemas import HealthResponse, VersionResponse
from app.services.detector import run_inference

app = FastAPI(title="CVision Box Intake API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        service_version=SERVICE_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        schema_version=SCHEMA_VERSION,
    )


@app.post("/v1/box-intake/infer")
async def infer(file: UploadFile = File(...)) -> dict:
    start_time = time.time()
    request_id = str(uuid.uuid4())
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Send JPEG or PNG."
        )
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(contents) > 10 * 1024 * 1024:
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
    detections = run_inference(image)
    detection_list = []
    size_summary = {"small": 0, "medium": 0, "large": 0}
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        size_class = (
            "small"
            if area_ratio < 0.03
            else ("large" if area_ratio > 0.10 else "medium")
        )
        size_summary[size_class] += 1
        touches_edge = x1 <= 5 or y1 <= 5 or x2 >= img_w - 5 or y2 >= img_h - 5
        detection_list.append(
            {
                "id": i + 1,
                "bbox_xyxy": d["bbox_xyxy"],
                "confidence": d["confidence"],
                "size_class": size_class,
                "area_ratio": round(area_ratio, 4),
                "touches_edge": touches_edge,
            }
        )
    review_reasons = []
    if len(detection_list) == 0:
        review_reasons.append("no_boxes_detected")
    if any(d["confidence"] < 0.45 for d in detection_list):
        review_reasons.append("low_confidence_detection")
    edge_count = sum(1 for d in detection_list if d["touches_edge"])
    if len(detection_list) > 0 and edge_count / len(detection_list) > 0.20:
        review_reasons.append("boxes_cut_off_at_edge")
    if len(detection_list) == 0:
        confidence_score = 0.0
    else:
        base = min(d["confidence"] for d in detection_list)
        penalties = 0.15 if any(d["confidence"] < 0.45 for d in detection_list) else 0.0
        if len(detection_list) > 0 and edge_count / len(detection_list) > 0.20:
            penalties += 0.05
        confidence_score = round(max(base - penalties, 0.05), 2)
    annotated = image.copy()
    for d in detection_list:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{d['size_class']} {d['confidence']:.2f}",
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
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": "box_intake_scan",
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": {"width": img_w, "height": img_h, "quality_flags": []},
        "detections": detection_list,
        "visible_box_count": len(detection_list),
        "size_summary": size_summary,
        "confidence_score": confidence_score,
        "human_review_required": len(review_reasons) > 0,
        "review_reasons": review_reasons,
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "service": {"version": SERVICE_VERSION},
        "processing_time_ms": round((time.time() - start_time) * 1000),
        "annotated_image_png_b64": annotated_b64,
    }
