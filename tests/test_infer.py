import json
from collections.abc import Iterator
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with (
        patch("app.main.assess_image_quality", return_value=[]),
        TestClient(app) as test_client,
    ):
        yield test_client


def test_health_returns_expected_contract(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_describes_service(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "CVision Box Intake API",
        "status": "ok",
        "health": "/health",
        "documentation": "/docs",
    }


def test_version_returns_model_and_schema_versions(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "service_version": "0.9.0",
        "model_name": "carton-yolov8n-truck",
        "model_version": "ft-truck-v1",
        "model_backend": "pytorch",
        "schema_version": "1.1",
    }


def test_infer_rejects_unsupported_file_type(client: TestClient) -> None:
    response = client.post(
        "/v1/box-intake/infer",
        files={"file": ("boxes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file type. Send JPEG or PNG."


def test_infer_rejects_corrupted_image(client: TestClient) -> None:
    response = client.post(
        "/v1/box-intake/infer",
        files={"file": ("boxes.jpg", b"not a decodable image", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == ("Cannot read image file. It may be corrupted.")


def test_infer_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/v1/box-intake/infer",
        files={"file": ("boxes.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file uploaded."


def test_infer_rejects_oversized_upload(client: TestClient) -> None:
    with patch("app.main.MAX_UPLOAD_BYTES", 4):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", b"12345", "image/jpeg")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image too large. Maximum size is 10MB."


def test_infer_returns_structured_count(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [{"bbox_xyxy": [10, 10, 50, 50], "confidence": 0.9}]

    with patch("app.main.run_inference", return_value=detections):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.1"
    assert body["visible_box_count"] == 1
    assert body["size_summary"] == {"small": 0, "medium": 0, "large": 1}
    assert body["human_review_required"] is False
    assert body["model"]["backend"] == "pytorch"
    assert response.headers["x-request-id"] == body["request_id"]


def test_infer_routes_zero_detections_to_review(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded

    with patch("app.main.run_inference", return_value=[]):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["visible_box_count"] == 0
    assert body["confidence_score"] == 0.0
    assert body["human_review_required"] is True
    assert "no_boxes_detected" in body["review_reasons"]


def test_infer_flags_low_confidence_detection(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [{"bbox_xyxy": [10, 10, 50, 50], "confidence": 0.4}]

    with patch("app.main.run_inference", return_value=detections):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert "low_confidence_detection" in response.json()["review_reasons"]


def test_infer_flags_edge_truncation(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [{"bbox_xyxy": [0, 10, 50, 50], "confidence": 0.9}]

    with patch("app.main.run_inference", return_value=detections):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert "boxes_cut_off_at_edge" in response.json()["review_reasons"]


def test_infer_does_not_flag_expected_high_detection_count(client: TestClient) -> None:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [
        {
            "bbox_xyxy": [10 + index * 2, 10, 20 + index * 2, 20],
            "confidence": 0.9,
        }
        for index in range(12)
    ]

    with (
        patch("app.main.assess_image_quality", return_value=[]),
        patch("app.main.run_inference", return_value=detections),
    ):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["visible_box_count"] == 12
    assert body["human_review_required"] is False
    assert "high_detection_count" not in body["review_reasons"]


def test_infer_reports_average_detection_confidence(client: TestClient) -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [
        {"bbox_xyxy": [10, 10, 40, 40], "confidence": 0.58},
        {"bbox_xyxy": [60, 10, 90, 40], "confidence": 0.92},
        {"bbox_xyxy": [110, 10, 140, 40], "confidence": 0.90},
    ]

    with (
        patch("app.main.assess_image_quality", return_value=[]),
        patch("app.main.run_inference", return_value=detections),
    ):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["confidence_score"] == 0.8


def test_infer_routes_poor_image_quality_to_review(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [{"bbox_xyxy": [10, 10, 50, 50], "confidence": 0.9}]

    with (
        patch("app.main.assess_image_quality", return_value=["possible_blur"]),
        patch("app.main.run_inference", return_value=detections),
    ):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["image"]["quality_flags"] == ["possible_blur"]
    assert body["human_review_required"] is True
    assert "poor_image_quality" in body["review_reasons"]
    assert body["confidence_score"] == 0.9


def test_infer_flags_possible_fragmented_detections(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [
        {"bbox_xyxy": [10, 10, 40, 70], "confidence": 0.9},
        {"bbox_xyxy": [42, 12, 72, 68], "confidence": 0.88},
    ]

    with patch("app.main.run_inference", return_value=detections):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["count_risk"]["suspicious_fragment_pair_count"] == 1
    assert "possible_fragmented_detections" in body["review_reasons"]


def test_infer_keeps_visual_review_advisory_without_changing_count(
    client: TestClient,
) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    detections = [{"bbox_xyxy": [10, 10, 50, 50], "confidence": 0.9}]
    assessment = {
        "status": "completed",
        "provider": "google_gemini",
        "model": "test-model",
        "visual_review_required": True,
        "visual_risk_reasons": ["reflection_or_shadow"],
        "summary": "A shadow resembles another carton.",
        "operator_guidance": "Check the shadowed region before accepting the count.",
        "novel_reason": None,
        "confidence": 0.8,
    }

    with (
        patch("app.main.run_inference", return_value=detections),
        patch("app.main.VISUAL_REVIEW_ENABLED", True),
        patch("app.main.assess_visual_review", return_value=assessment),
    ):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["visible_box_count"] == 1
    assert body["human_review_required"] is False
    assert body["review_reasons"] == []
    assert body["review_assessment"] == assessment


def test_infer_routes_and_logs_novel_visual_risk(client: TestClient) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    assessment = {
        "status": "completed",
        "provider": "google_gemini",
        "model": "test-model",
        "visual_review_required": True,
        "visual_risk_reasons": ["other_visual_risk"],
        "summary": "A loose strap obscures carton boundaries.",
        "operator_guidance": "Move the strap and retake the image.",
        "novel_reason": "A loose packing strap crosses multiple carton faces.",
        "confidence": 0.79,
    }

    with (
        patch("app.main.run_inference", return_value=[]),
        patch("app.main.VISUAL_REVIEW_ENABLED", True),
        patch("app.main.assess_visual_review", return_value=assessment),
        patch("app.main.logger.warning") as warning,
    ):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    body = response.json()
    assert body["human_review_required"] is True
    assert body["review_reasons"] == ["no_boxes_detected"]
    assert body["review_assessment"]["visual_risk_reasons"] == ["other_visual_risk"]
    log_payload = json.loads(warning.call_args.args[1])
    assert log_payload["event"] == "novel_visual_risk_detected"
    assert log_payload["request_id"] == body["request_id"]
    assert log_payload["novel_reason"] == assessment["novel_reason"]


def test_infer_maps_model_failure_to_service_unavailable(
    client: TestClient,
) -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded

    with patch("app.main.run_inference", side_effect=RuntimeError("runtime failed")):
        response = client.post(
            "/v1/box-intake/infer",
            files={"file": ("boxes.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model inference is temporarily unavailable."
