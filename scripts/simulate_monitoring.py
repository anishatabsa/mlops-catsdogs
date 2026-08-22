"""
M5 - Post-deployment model performance tracking.

Sends a batch of real (test-split) or simulated images with known true
labels to the running inference service, compares predictions against
ground truth, and reports accuracy/precision/recall - i.e. a lightweight
"model performance in production" report you'd otherwise get from a
monitoring dashboard.

Usage:
    python scripts/simulate_monitoring.py \
        --api-url http://localhost:8000 \
        --data-dir data/processed/test \
        --n-per-class 25
"""
import argparse
import json
import random
import time
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def sample_images(data_dir: str, n_per_class: int, seed: int = 42):
    root = Path(data_dir)
    rng = random.Random(seed)
    samples = []
    for class_name in ["cat", "dog"]:
        class_dir = root / class_name
        files = sorted(class_dir.glob("*"))
        rng.shuffle(files)
        for f in files[:n_per_class]:
            samples.append((f, class_name))
    rng.shuffle(samples)
    return samples


def run_batch(api_url: str, samples) -> dict:
    y_true, y_pred, latencies = [], [], []
    for path, true_label in samples:
        with open(path, "rb") as f:
            start = time.time()
            resp = requests.post(f"{api_url}/predict", files={"file": (path.name, f, "image/jpeg")})
            latencies.append((time.time() - start) * 1000)
        resp.raise_for_status()
        body = resp.json()
        y_true.append(true_label)
        y_pred.append(body["label"])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "latencies_ms": latencies,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--data-dir", default="data/processed/test")
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--output", default="artifacts/monitoring_report.json")
    args = parser.parse_args()

    samples = sample_images(args.data_dir, args.n_per_class)
    print(f"Sending {len(samples)} requests to {args.api_url}/predict ...")
    result = run_batch(args.api_url, samples)

    acc = accuracy_score(result["y_true"], result["y_pred"])
    report = classification_report(result["y_true"], result["y_pred"], output_dict=True)
    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=["cat", "dog"]).tolist()
    avg_latency = sum(result["latencies_ms"]) / len(result["latencies_ms"])

    summary = {
        "n_requests": len(samples),
        "accuracy": acc,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(sorted(result["latencies_ms"])[int(0.95 * len(result["latencies_ms"])) - 1], 2),
        "classification_report": report,
        "confusion_matrix": cm,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
