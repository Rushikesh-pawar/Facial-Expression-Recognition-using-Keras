"""Central configuration for training, data, and inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# FER-2013 emotion class labels in dataset order.
EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# Hex colors used for bounding-box overlays per class.
EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "angry":    (60,  60,  220),   # red
    "disgust":  (60,  180, 75),    # green
    "fear":     (170, 110, 40),    # brown
    "happy":    (25,  200, 245),   # yellow
    "sad":      (200, 130, 0),     # blue
    "surprise": (240, 50,  230),   # magenta
    "neutral":  (200, 200, 200),   # gray
}


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and paths for a training run."""

    image_size: int = 224
    batch_size: int = 64
    num_classes: int = 7
    seed: int = 42

    # Stage 1: train classifier head with frozen backbone.
    head_epochs: int = 10
    head_lr: float = 1e-3

    # Stage 2: unfreeze top layers and fine-tune.
    finetune_epochs: int = 25
    finetune_lr: float = 1e-4
    finetune_unfreeze_from_layer: int = 100

    label_smoothing: float = 0.1
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    use_class_weights: bool = True
    early_stopping_patience: int = 6

    augmentation: dict = field(
        default_factory=lambda: {
            "horizontal_flip": True,
            "rotation": 0.08,
            "zoom": 0.1,
            "contrast": 0.15,
            "brightness": 0.1,
        }
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime parameters for the webcam / web inference loop."""

    image_size: int = 224
    detection_confidence: float = 0.5
    smoothing_window: int = 5
    camera_index: int = 0
    fps_target: int = 30
