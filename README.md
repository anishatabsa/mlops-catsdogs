# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classification (cats vs dogs) for a pet-adoption platform, built as a full
MLOps pipeline: data & model versioning, experiment tracking, packaging, containerization,
CI, CD, and monitoring — using only open-source tooling.

| Milestone | What it covers | Where |
|---|---|---|
| M1 | Data/code versioning, baseline CNN, experiment tracking | `src/data`, `src/models`, DVC, MLflow |
| M2 | FastAPI inference service, pinned deps, Docker | `src/api`, `docker/` |
| M3 | Unit tests, CI (GitHub Actions), image publishing | `tests/`, `.github/workflows/ci.yml` |
| M4 | Docker Compose deployment, CD, smoke tests | `docker-compose.yml`, `.github/workflows/cd.yml` |
| M5 | Logging, metrics, post-deploy performance tracking | `src/monitoring`, `scripts/simulate_monitoring.py` |

## Repository layout

```
src/
  data/            # download + preprocessing (resize/split) — M1
  models/          # CNN, dataset/augmentation, train loop (MLflow), inference — M1/M2
  api/             # FastAPI service: /health /predict /metrics — M2
  monitoring/      # structured logging + Prometheus metrics — M5
tests/             # pytest unit + API tests — M3
docker/            # Dockerfile, .dockerignore — M2
docker-compose.yml # deployment target — M4
monitoring/        # prometheus.yml scrape config — M5
.github/workflows/ # ci.yml (test/build/push), cd.yml (deploy/smoke test) — M3/M4
scripts/           # smoke_test.sh, simulate_monitoring.py — M4/M5
dvc.yaml, .dvc/    # data + pipeline versioning — M1
mlruns/            # local MLflow tracking store — M1
data/              # raw/ + processed/ (DVC-tracked, not in git) — M1
models/model.pt    # trained checkpoint shipped for grading — M1
docs/architecture.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### M1 — Data + training + experiment tracking

```bash
# 1. Get the data (see data/README.md for Kaggle API setup)
python -m src.data.download_data --output data/raw

# 2. Pre-process: resize to 224x224 RGB, stratified 80/10/10 split
python -m src.data.preprocess --raw-dir data/raw --processed-dir data/processed

# 3. Version the data with DVC
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc
git commit -m "Track datasets with DVC"

# 4. Train the baseline CNN, tracked with MLflow
python -m src.models.train --data-dir data/processed --epochs 5 --output models/model.pt

# 5. Inspect the run (params, metrics, confusion matrix, loss curve, model artifact)
mlflow ui --backend-store-uri mlruns --port 5000
```

### M2 — Serve + containerize

```bash
# Run locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict -F "file=@data/processed/test/cat/00000.jpg"

# Build & run the container
docker build -f docker/Dockerfile -t catsdogs-inference:latest .
docker run --rm -p 8000:8000 catsdogs-inference:latest
```

### M3 — Tests + CI

```bash
pytest -v --cov=src tests/
```

Push to any branch (or open a PR into `main`) and GitHub Actions (`.github/workflows/ci.yml`)
will check out the repo, install dependencies, run the test suite, build the Docker image, and
— on pushes — publish it to `ghcr.io/<your-repo>`.

### M4 — Deploy + smoke test

```bash
docker compose up -d --build
bash scripts/smoke_test.sh http://localhost:8000
```

On a push to `main`, once CI has published a new image, `.github/workflows/cd.yml` pulls it,
redeploys with Docker Compose, waits for `/health`, and runs the same smoke test — failing
(and rolling back) the deployment if it doesn't pass.

### M5 — Monitoring

- Every `/predict` call is logged as one structured JSON line (filename, content-type, size,
  predicted label, confidence, latency — **no raw image bytes**), and counted/timed via
  Prometheus metrics exposed at `GET /metrics`.
- `docker-compose.yml` also brings up a Prometheus instance (`localhost:9090`) scraping those
  metrics.
- To simulate post-deployment traffic and get an accuracy/latency report against known labels:

  ```bash
  python scripts/simulate_monitoring.py --api-url http://localhost:8000 \
      --data-dir data/processed/test --n-per-class 25
  ```

  This writes `artifacts/monitoring_report.json` with accuracy, precision/recall,
  confusion matrix, and p95 latency for the sampled batch.

## Model card (baseline)

- Architecture: 4-block CNN (`src/models/model.py`, `SimpleCNN`) — conv/batchnorm/relu/maxpool
  x4 -> global average pool -> dropout -> linear(2)
- Input: 224x224 RGB, ImageNet-style normalization
- Alternate baseline logged for comparison: logistic regression on flattened 32x32 pixels
  (`src/models/baseline_logreg.py`)
- Training/validation metrics, the confusion matrix, and the loss curve for every run are in
  MLflow (`mlruns/`) — see `mlflow ui` above.

## Screen recording

This repo includes everything needed to demonstrate the full flow (code change -> CI ->
image published -> CD deploy -> live prediction). A suggested under-5-minute recording script
is in `docs/demo_script.md`.
