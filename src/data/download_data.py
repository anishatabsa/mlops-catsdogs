"""
Download the Cats vs Dogs classification dataset from Kaggle.

Requires Kaggle API credentials to be available either as:
  - environment variables KAGGLE_USERNAME and KAGGLE_KEY, or
  - a kaggle.json file at ~/.kaggle/kaggle.json (chmod 600)

Dataset used: tongpython/cat-and-dog (a curated Kaggle mirror of the classic
Cats vs Dogs binary classification dataset - two folders, `cats` and `dogs`,
containing labeled JPEG images). Any similarly-structured Kaggle cats/dogs
dataset can be substituted by changing KAGGLE_DATASET below.

Usage:
    python -m src.data.download_data --output data/raw
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "tongpython/cat-and-dog"


def ensure_kaggle_credentials() -> None:
    """Verify Kaggle API credentials are available, raise a clear error otherwise."""
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_file):
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY env vars, "
            "or place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)."
        )


def download(output_dir: str, dataset: str = KAGGLE_DATASET) -> Path:
    ensure_kaggle_credentials()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset,
        "-p", str(out),
        "--unzip",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Some Kaggle mirrors nest an extra zip or directory level - flatten if needed.
    for extra_zip in out.glob("*.zip"):
        with zipfile.ZipFile(extra_zip) as zf:
            zf.extractall(out)
        extra_zip.unlink()

    print(f"Dataset downloaded and extracted to {out.resolve()}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw", help="Output directory for raw data")
    parser.add_argument("--dataset", default=KAGGLE_DATASET, help="Kaggle dataset slug")
    args = parser.parse_args()
    download(args.output, args.dataset)
