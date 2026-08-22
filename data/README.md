# Data

This directory holds the Cats vs Dogs dataset at two stages of the pipeline:

- `raw/` — the original Kaggle download, laid out as `raw/cats/*.jpg` and `raw/dogs/*.jpg`
  (or `raw/training_set/...`, `raw/test_set/...` depending on which Kaggle mirror you use —
  `src/data/preprocess.py` searches common aliases automatically).
- `processed/` — output of `src/data/preprocess.py`: every image resized to 224x224 RGB and
  split into `processed/{train,val,test}/{cat,dog}/`.

Neither directory is committed to git (see `.gitignore`) — both are tracked with **DVC**
instead, so large binary data never bloats the git history while still being versioned
alongside the code that produced it.

## Getting the data

**Option A — Kaggle API (recommended, gets the full dataset):**

```bash
pip install kaggle
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

python -m src.data.download_data --output data/raw
```

**Option B — manual download:** grab the dataset zip from Kaggle in your browser and unzip
it into `data/raw/`.

## Pre-processing + versioning with DVC

```bash
# Resize to 224x224 RGB and split 80/10/10 (stratified by class)
python -m src.data.preprocess --raw-dir data/raw --processed-dir data/processed

# Track both stages with DVC
dvc add data/raw
dvc add data/processed
git add data/raw.dvc data/processed.dvc data/.gitignore
git commit -m "Track raw and processed datasets with DVC"

# Push the actual data to the configured DVC remote (see .dvc/config)
dvc push
```

Anyone who clones the repo can then reproduce the exact same dataset with:

```bash
dvc pull
```

Or replay the whole pipeline (preprocess -> train) with `dvc repro`, per `dvc.yaml`.
