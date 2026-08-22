"""
Integration-style tests for the FastAPI service using TestClient.
Requires a model checkpoint at models/model.pt (created by conftest.py
if missing, using an untrained model so CI doesn't need the dataset).
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


@pytest.fixture
def client():
    # Using TestClient as a context manager triggers FastAPI's startup
    # event (model loading) before the tests run, and shutdown after.
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["model_loaded"] is True


def test_predict_endpoint_returns_prediction(client):
    img = Image.new("RGB", (300, 300), color=(10, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"cat", "dog"}
    assert "confidence" in body
    assert "latency_ms" in body


def test_predict_endpoint_rejects_bad_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inference_requests_total" in response.text or response.text == ""
