"""
Structured logging + in-app metrics for the inference service (M5).

- Request/response logging: one JSON line per prediction request, excluding
  the raw image bytes (only shape/size + result are logged - no sensitive
  payload data).
- Metrics: request counters and latency histograms exposed via
  prometheus_client, scraped at GET /metrics.
"""
import json
import logging
import sys
import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

logger = logging.getLogger("inference_service")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

# --- Prometheus metrics -----------------------------------------------------
REQUEST_COUNT = Counter(
    "inference_requests_total", "Total number of prediction requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions by predicted label", ["label"]
)


def log_event(event: str, **fields):
    """Emit a single structured JSON log line. Never pass raw image bytes in fields."""
    record = {"event": event, "timestamp": time.time(), **fields}
    logger.info(json.dumps(record))


@contextmanager
def track_request(endpoint: str):
    """Context manager: times a request, records Prometheus metrics + a log line."""
    start = time.time()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        log_event("request", endpoint=endpoint, status=status, latency_ms=round(elapsed * 1000, 2))
