"""
Optional even-simpler baseline: logistic regression on flattened,
downsampled pixel values. Not used by the API (the CNN is the shipped
model), but included to satisfy "at least one baseline model (e.g.
simple CNN OR logistic regression on flattened pixels)" with a second
point of comparison, and logged to MLflow as its own run for reference.
"""
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

FLAT_SIZE = 32  # downsample to 32x32 RGB -> 3072 features, keeps this fast


def image_to_flat_vector(path, size: int = FLAT_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def build_dataset(items, size: int = FLAT_SIZE):
    X = np.stack([image_to_flat_vector(p, size) for p, _ in items])
    y = np.array([1 if label == "dog" else 0 for _, label in items])
    return X, y


def train_logreg(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    metrics = {
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred),
    }
    return clf, metrics
