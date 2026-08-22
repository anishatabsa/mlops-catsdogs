#!/usr/bin/env bash
# Post-deploy smoke test: calls the health endpoint and one prediction call.
# Fails (non-zero exit) if either check fails, so a CD pipeline can gate on it.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-scripts/sample_test_image.jpg}"

echo "== Smoke test against ${BASE_URL} =="

echo "-- Checking /health"
HEALTH_RESPONSE=$(curl -sf "${BASE_URL}/health")
echo "${HEALTH_RESPONSE}"
echo "${HEALTH_RESPONSE}" | grep -q '"model_loaded":true' \
  || { echo "FAIL: model not loaded"; exit 1; }

if [ ! -f "${SAMPLE_IMAGE}" ]; then
  echo "-- No sample image found at ${SAMPLE_IMAGE}, generating a synthetic one"
  python3 - "$SAMPLE_IMAGE" <<'EOF'
import sys
from PIL import Image
Image.new("RGB", (224, 224), color=(128, 90, 60)).save(sys.argv[1])
EOF
fi

echo "-- Checking /predict"
PREDICT_RESPONSE=$(curl -sf -X POST "${BASE_URL}/predict" -F "file=@${SAMPLE_IMAGE};type=image/jpeg")
echo "${PREDICT_RESPONSE}"
echo "${PREDICT_RESPONSE}" | grep -q '"label"' \
  || { echo "FAIL: /predict did not return a label"; exit 1; }

echo "== Smoke test PASSED =="
