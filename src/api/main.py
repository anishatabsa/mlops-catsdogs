"""
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   - liveness/readiness probe
    POST /predict  - accepts an image file, returns class probabilities/label
    GET  /metrics  - Prometheus-format metrics (M5 monitoring)

The model is loaded once at process startup and reused across requests.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse, PredictResponse
from src.models.predict import load_model, predict_image
from src.monitoring.logging_config import PREDICTION_COUNT, log_event, track_request

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pt")

_model = None
_model_load_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_load_error
    try:
        _model = load_model(MODEL_PATH)
        log_event("model_loaded", model_path=MODEL_PATH)
    except Exception as e:  # noqa: BLE001 - we want health to report this, not crash startup
        _model_load_error = str(e)
        log_event("model_load_failed", error=_model_load_error)
    yield


app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if _model is not None else "degraded",
                           model_loaded=_model is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error}")

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    with track_request("predict"):
        start = time.time()
        image_bytes = await file.read()
        try:
            result = predict_image(_model, image_bytes)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

        latency_ms = round((time.time() - start) * 1000, 2)
        PREDICTION_COUNT.labels(label=result["label"]).inc()

        # Log request/response WITHOUT the raw image bytes - only metadata + result.
        log_event(
            "prediction",
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=len(image_bytes),
            predicted_label=result["label"],
            confidence=round(result["confidence"], 4),
            latency_ms=latency_ms,
        )

        return PredictResponse(
            label=result["label"],
            probabilities=result["probabilities"],
            confidence=result["confidence"],
            latency_ms=latency_ms,
        )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
