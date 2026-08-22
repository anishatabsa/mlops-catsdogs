"""
Unit tests for the model inference utility (src/models/predict.py).

Covers the "one model utility/inference function" requirement (M3).
Uses a freshly-initialized (untrained) model so the tests run fast and
don't depend on a downloaded dataset or a trained checkpoint - they
verify the *contract* of predict_image(), not model accuracy.
"""
import io

import pytest
from PIL import Image

from src.models.model import CLASS_NAMES, build_model
from src.models.predict import _to_pil, predict_image


@pytest.fixture
def untrained_model():
    model = build_model()
    model.eval()
    return model


@pytest.fixture
def sample_pil_image():
    return Image.new("RGB", (300, 250), color=(120, 60, 200))


def test_predict_image_returns_expected_keys(untrained_model, sample_pil_image):
    result = predict_image(untrained_model, sample_pil_image)
    assert set(result.keys()) == {"label", "probabilities", "confidence"}


def test_predict_image_label_is_valid_class(untrained_model, sample_pil_image):
    result = predict_image(untrained_model, sample_pil_image)
    assert result["label"] in CLASS_NAMES


def test_predict_image_probabilities_sum_to_one(untrained_model, sample_pil_image):
    result = predict_image(untrained_model, sample_pil_image)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_image_confidence_matches_label_probability(untrained_model, sample_pil_image):
    result = predict_image(untrained_model, sample_pil_image)
    assert result["confidence"] == pytest.approx(result["probabilities"][result["label"]])


def test_predict_image_accepts_raw_bytes(untrained_model, sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    result = predict_image(untrained_model, raw_bytes)
    assert result["label"] in CLASS_NAMES


def test_to_pil_converts_bytes_to_rgb_image(sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    pil = _to_pil(buf.getvalue())
    assert pil.mode == "RGB"


def test_to_pil_rejects_unsupported_type():
    with pytest.raises(TypeError):
        _to_pil(12345)
