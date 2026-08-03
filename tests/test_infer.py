from collections.abc import Iterator
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_expected_contract(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_returns_model_and_schema_versions(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "service_version": "0.1.0",
        "model_name": "carton-yolov8n",
        "model_version": "ft-v1",
        "schema_version": "1.0",
    }


def test_infer_rejects_unsupported_file_type(client: TestClient) -> None:
    response = client.post(
        "/v1/box-intake/infer",
        files={"file": ("boxes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file type. Send JPEG or PNG."


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
    assert body["schema_version"] == "1.0"
    assert body["visible_box_count"] == 1
    assert body["size_summary"] == {"small": 0, "medium": 0, "large": 1}
    assert body["human_review_required"] is False
