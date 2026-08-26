"""
Data pre-processing for the Cats vs Dogs binary classification pipeline.

Responsibilities:
  1. Scan a raw dataset directory laid out as <raw_dir>/{Cat,Dog|cats,dogs}/*.jpg
  2. Resize every valid image to 224x224 RGB (standard CNN input size)
  3. Stratified split into train/val/test (default 80/10/10)
  4. Write the processed images to <processed_dir>/{train,val,test}/{cat,dog}/

Data augmentation is intentionally NOT baked into the saved files - it is
applied on-the-fly at training time (see src/models/dataset.py) so that
every epoch sees fresh augmented variants of the same underlying image.
"""
import argparse
import random
import warnings
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

IMG_SIZE = (224, 224)
CLASSES = ["cat", "dog"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Different Kaggle mirrors use different folder casing/naming for the same classes.
CLASS_DIR_ALIASES = {
    "cat": ["cat", "cats", "Cat", "Cats", "PetImages/Cat"],
    "dog": ["dog", "dogs", "Dog", "Dogs", "PetImages/Dog"],
}


def _count_images(d: Path) -> int:
    return sum(1 for f in d.glob("*") if f.suffix.lower() in IMAGE_EXTS)


def _find_class_dir(raw_dir: Path, class_name: str) -> Path:
    """
    Locate the directory holding this class's images.

    Collects every direct-child and nested (one-or-more-levels-deep) alias
    match and picks the one with the MOST image files, rather than the
    first alias that happens to match. This matters because a raw_dir can
    contain more than one plausible candidate at once - e.g. a small
    leftover/sample folder alongside the real, much larger dataset - and
    silently picking the first match would train on the wrong data without
    any error. Warns when more than one non-trivial candidate is found.
    """
    candidates = []
    seen = set()
    for alias in CLASS_DIR_ALIASES[class_name]:
        for candidate in [raw_dir / alias, *raw_dir.rglob(alias)]:
            if candidate.is_dir() and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a '{class_name}' image directory under {raw_dir}. "
            f"Tried aliases: {CLASS_DIR_ALIASES[class_name]}"
        )

    counts = [(c, _count_images(c)) for c in candidates]
    counts.sort(key=lambda pair: pair[1], reverse=True)
    best_dir, best_count = counts[0]

    non_trivial = [c for c in counts if c[1] > 0]
    if len(non_trivial) > 1:
        warnings.warn(
            f"Multiple '{class_name}' image directories found under {raw_dir}: "
            + ", ".join(f"{c} ({n} images)" for c, n in counts)
            + f". Using the largest: {best_dir} ({best_count} images). "
            "If that's not the one you meant, remove or rename the others."
        )

    return best_dir


def scan_dataset(raw_dir: str) -> List[Tuple[Path, str]]:
    """Return a list of (image_path, class_name) tuples found under raw_dir."""
    raw_path = Path(raw_dir)
    items: List[Tuple[Path, str]] = []
    for class_name in CLASSES:
        class_dir = _find_class_dir(raw_path, class_name)
        for f in sorted(class_dir.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((f, class_name))
    if not items:
        raise RuntimeError(f"No images found under {raw_dir}")
    return items


def resize_image(image_path, size: Tuple[int, int] = IMG_SIZE) -> Image.Image:
    """
    Load an image from disk and return it resized to `size` in RGB mode.

    This is the core, independently-testable pre-processing function:
    given any input image (any mode, any resolution), it must always
    return a 224x224 RGB PIL Image.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        img.load()
    return img


def is_valid_image(image_path) -> bool:
    """Return True if the file at image_path can be opened as an image."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def stratified_split(
    items: List[Tuple[Path, str]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
):
    """Split items into train/val/test, stratified by class, deterministically."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"

    rng = random.Random(seed)
    by_class = {c: [i for i in items if i[1] == c] for c in CLASSES}

    train, val, test = [], [], []
    for c, class_items in by_class.items():
        class_items = class_items[:]
        rng.shuffle(class_items)
        n = len(class_items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += class_items[:n_train]
        val += class_items[n_train:n_train + n_val]
        test += class_items[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_processed_dataset(
    raw_dir: str,
    processed_dir: str,
    size: Tuple[int, int] = IMG_SIZE,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    limit_per_class: int = None,
    skip_invalid: bool = True,
) -> dict:
    """
    Full pre-processing pipeline: scan -> validate -> split -> resize -> save.
    Returns a summary dict with per-split, per-class counts.
    """
    items = scan_dataset(raw_dir)

    if skip_invalid:
        items = [i for i in items if is_valid_image(i[0])]

    if limit_per_class:
        capped = []
        for c in CLASSES:
            capped += [i for i in items if i[1] == c][:limit_per_class]
        items = capped

    train, val, test = stratified_split(
        items, train_frac, val_frac, test_frac, seed=seed
    )

    summary = {}
    out_root = Path(processed_dir)
    for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
        counts = {c: 0 for c in CLASSES}
        for src_path, class_name in split_items:
            dest_dir = out_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{counts[class_name]:05d}{src_path.suffix.lower()}"
            try:
                img = resize_image(src_path, size)
                img.save(dest_path, format="JPEG", quality=90)
                counts[class_name] += 1
            except (UnidentifiedImageError, OSError):
                continue
        summary[split_name] = counts

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Optional cap per class, useful for fast local iteration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_processed_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        size=(args.size, args.size),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        limit_per_class=args.limit_per_class,
    )
    print("Processed dataset summary:")
    for split_name, counts in result.items():
        print(f"  {split_name}: {counts}")
