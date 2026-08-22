.PHONY: install preprocess train serve test docker-build docker-run compose-up compose-down smoke monitor lint

VENV ?= .venv
PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt

preprocess:
	$(PY) -m src.data.preprocess --raw-dir data/raw --processed-dir data/processed

train:
	$(PY) -m src.models.train --data-dir data/processed --epochs 5 --output models/model.pt

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v --cov=src --cov-report=term-missing tests/

docker-build:
	docker build -f docker/Dockerfile -t catsdogs-inference:latest .

docker-run:
	docker run --rm -p 8000:8000 catsdogs-inference:latest

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

smoke:
	bash scripts/smoke_test.sh http://localhost:8000

monitor:
	$(PY) scripts/simulate_monitoring.py --api-url http://localhost:8000 --data-dir data/processed/test --n-per-class 25

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000
