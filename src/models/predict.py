"""
Inference utility: load a trained model checkpoint and run predictions on
raw images (bytes, file path, or PIL Image). Shared by the FastAPI service
and by the unit tests, so training and serving never drift apart on
pre-processing logic.
"""
import io
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src.models.dataset import eval_transforms
from src.models.model import CLASS_NAMES, build_model

DEFAULT_MODEL_PATH = "models/model.pt"


def load_model(model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu") -> torch.nn.Module:
    """Load a SimpleCNN checkpoint saved via torch.save(model.state_dict(), path)."""
    model = build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def _to_pil(image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)}")


@torch.no_grad()
def predict_image(model: torch.nn.Module, image: Union[str, Path, bytes, Image.Image], device: str = "cpu") -> dict:
    """
    Run a single-image prediction.

    Returns:
        {
          "label": "cat" | "dog",
          "probabilities": {"cat": float, "dog": float},
          "confidence": float,
        }
    """
    pil_img = _to_pil(image)
    tensor = eval_transforms()(pil_img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    label = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "label": label,
        "probabilities": prob_dict,
        "confidence": prob_dict[label],
    }
