"""
Baseline CNN for binary Cats vs Dogs classification.

Kept intentionally small (a handful of conv blocks) so it trains in a
reasonable time on CPU while still being a genuine CNN baseline (as
opposed to logistic regression on flattened pixels, which is offered
as an even-simpler baseline in baseline_logreg.py).
"""
import torch
import torch.nn as nn

NUM_CLASSES = 2
CLASS_NAMES = ["cat", "dog"]


class SimpleCNN(nn.Module):
    """A small CNN: 4 conv blocks + global average pool + linear head.

    Input: (B, 3, 224, 224) RGB image tensor, normalized.
    Output: (B, 2) raw logits over [cat, dog].
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int = NUM_CLASSES) -> SimpleCNN:
    return SimpleCNN(num_classes=num_classes)
