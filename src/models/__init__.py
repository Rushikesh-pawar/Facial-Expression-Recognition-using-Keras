"""Model architectures."""
from src.models.emotion_classifier import build_emotion_classifier, unfreeze_top_layers

__all__ = ["build_emotion_classifier", "unfreeze_top_layers"]
