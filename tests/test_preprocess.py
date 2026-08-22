"""
Unit tests for the data pre-processing function (src/data/preprocess.py).

Covers the "at least one data pre-processing function" requirement (M3):
resize_image() must always return a 224x224 RGB image regardless of the
input image's original size, mode, or format.
"""
import io

import pytest
from PIL import Image

from src.data.preprocess import (
    IMG_SIZE,
    is_valid_image,
    resize_image,
    stratified_split,
)


@pytest.fixture
def tmp_image(tmp_path):
    """Create a small non-square grayscale JPEG on disk for testing."""
    img = Image.new("L", (50, 80), color=128)  # grayscale, non-square, small
    path = tmp_path / "sample.jpg"
    img.save(path, format="JPEG")
    return path


def test_resize_image_output_size(tmp_image):
    result = resize_image(tmp_image)
    assert result.size == IMG_SIZE


def test_resize_image_output_is_rgb(tmp_image):
    result = resize_image(tmp_image)
    assert result.mode == "RGB"


def test_resize_image_with_custom_size(tmp_image):
    result = resize_image(tmp_image, size=(64, 64))
    assert result.size == (64, 64)


def test_resize_image_handles_rgba_input(tmp_path):
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    path = tmp_path / "rgba.png"
    img.save(path, format="PNG")

    result = resize_image(path)
    assert result.mode == "RGB"
    assert result.size == IMG_SIZE


def test_is_valid_image_true_for_real_image(tmp_image):
    assert is_valid_image(tmp_image) is True


def test_is_valid_image_false_for_corrupt_file(tmp_path):
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"this is not a real jpeg file")
    assert is_valid_image(bad_path) is False


def test_stratified_split_proportions_and_no_overlap(tmp_path):
    # Build a fake item list: 100 cats + 100 dogs
    items = [(tmp_path / f"cat_{i}.jpg", "cat") for i in range(100)]
    items += [(tmp_path / f"dog_{i}.jpg", "dog") for i in range(100)]

    train, val, test = stratified_split(items, 0.8, 0.1, 0.1, seed=42)

    assert len(train) == 160  # 80 cats + 80 dogs
    assert len(val) == 20
    assert len(test) == 20

    # No file should appear in more than one split
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_stratified_split_is_deterministic(tmp_path):
    items = [(tmp_path / f"cat_{i}.jpg", "cat") for i in range(20)]
    items += [(tmp_path / f"dog_{i}.jpg", "dog") for i in range(20)]

    split_a = stratified_split(items, seed=7)
    split_b = stratified_split(items, seed=7)
    assert split_a == split_b
