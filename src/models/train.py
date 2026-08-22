"""
Train the baseline CNN on the processed Cats vs Dogs dataset, tracking the
run (params, per-epoch metrics, confusion matrix, loss curve, and the
model artifact itself) with MLflow.

Usage:
    python -m src.models.train \
        --data-dir data/processed \
        --epochs 5 --batch-size 32 --lr 1e-3 \
        --output models/model.pt \
        --experiment cats-vs-dogs
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.dataset import load_image_folder
from src.models.model import CLASS_NAMES, build_model


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    f1 = f1_score(all_labels, all_preds, zero_division=0) if n else 0.0
    return avg_loss, acc, f1, all_labels, all_preds


def plot_loss_curve(train_losses, val_losses, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(labels, preds, out_path: str):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (val)")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/model.pt")
    parser.add_argument("--experiment", default="cats-vs-dogs")
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    train_ds = load_image_folder(str(Path(args.data_dir) / "train"), train=True)
    val_ds = load_image_folder(str(Path(args.data_dir) / "val"), train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run(run_name=f"cnn-{int(time.time())}") as run:
        mlflow.log_params({
            "model": "SimpleCNN",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "device": device,
        })

        train_losses, val_losses = [], []
        best_val_acc = 0.0
        last_labels, last_preds = [], []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = running_loss / max(n, 1)

            val_loss, val_acc, val_f1, labels, preds = evaluate(model, val_loader, device, criterion)
            last_labels, last_preds = labels, preds

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            }, step=epoch)

            print(f"epoch {epoch}/{args.epochs} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            best_val_acc = max(best_val_acc, val_acc)

        # Artifacts: loss curve + confusion matrix
        Path("artifacts").mkdir(exist_ok=True)
        loss_curve_path = "artifacts/loss_curve.png"
        cm_path = "artifacts/confusion_matrix.png"
        plot_loss_curve(train_losses, val_losses, loss_curve_path)
        plot_confusion_matrix(last_labels, last_preds, cm_path)
        mlflow.log_artifact(loss_curve_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Save + log model
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"Saved model to {out_path}, MLflow run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
