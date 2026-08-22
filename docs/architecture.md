# Architecture

## End-to-end flow

```
 Kaggle dataset
       |
       v
 [M1] preprocess.py --> data/processed/{train,val,test} (versioned with DVC)
       |
       v
 [M1] train.py --------> models/model.pt  (tracked as an MLflow run:
       |                  params, per-epoch metrics, confusion matrix,
       |                  loss curve, model artifact)
       v
 [M2] FastAPI service (src/api/main.py) wraps model.pt
       |    /health   /predict   /metrics
       v
 [M2/M3] Dockerfile builds a container image around the service
       |
       v
 [M3] GitHub Actions CI (.github/workflows/ci.yml)
       |    checkout -> install -> pytest -> docker build -> push to GHCR
       v
 [M4] GitHub Actions CD (.github/workflows/cd.yml)
       |    pull image -> docker compose up -> smoke test -> rollback on failure
       v
 [M4] docker-compose.yml running service (deployment target)
       |
       v
 [M5] Prometheus scrapes /metrics; structured JSON request logs;
      scripts/simulate_monitoring.py sends a labeled batch and reports
      post-deployment accuracy/latency
```

## Component responsibilities

- **src/data** — dataset download + pre-processing + train/val/test split (M1)
- **src/models** — CNN definition, dataset/augmentation pipeline, training loop with MLflow
  logging, inference utility shared by the API and the test suite (M1/M2)
- **src/api** — FastAPI inference service: health check, prediction, Prometheus metrics (M2)
- **src/monitoring** — structured logging + Prometheus counters/histograms (M5)
- **docker/** — Dockerfile + .dockerignore for the inference service (M2)
- **docker-compose.yml** — deployment target: inference service + Prometheus (M4)
- **.github/workflows** — CI (test, build, publish image) and CD (deploy, smoke test) (M3/M4)
- **scripts/** — smoke test and post-deploy monitoring/performance scripts (M4/M5)
- **dvc.yaml / .dvc** — data + pipeline versioning (M1)
- **mlruns/** — local MLflow tracking store (experiment runs, metrics, artifacts) (M1)

## Design choices and why

- **SimpleCNN over a pretrained backbone**: keeps training fast on CPU for grading/demo
  purposes while still being a genuine CNN (not just flattened-pixel logistic regression).
  `src/models/baseline_logreg.py` provides the even-simpler baseline as a second MLflow-logged
  comparison point.
- **Docker Compose over Kubernetes**: the assignment allows either; Compose was chosen for a
  simpler, fully local deployment story that's easy to demonstrate and reproduce without a
  cluster tool.
- **GHCR over Docker Hub**: no separate account/token needed — CI authenticates with the
  built-in `GITHUB_TOKEN`.
- **In-app Prometheus metrics + structured JSON logs over a heavier observability stack**:
  sufficient for "basic monitoring" (request count, latency, prediction distribution) without
  extra infrastructure, while remaining a real Prometheus-scrapeable `/metrics` endpoint.
