"""
Pytest fixtures shared across the test suite.

CI environments won't have a trained model.pt checkpoint (that requires
the full dataset + training run), so we transparently create one from an
untrained model if it's missing. This keeps unit/integration tests fast
and dataset-independent, while still exercising the full load -> predict
code path end to end.
"""
import os
from pathlib import Path

import torch

from src.models.model import build_model

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pt"))


def pytest_configure(config):
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = build_model()
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[conftest] Created placeholder untrained model at {MODEL_PATH} for CI")
