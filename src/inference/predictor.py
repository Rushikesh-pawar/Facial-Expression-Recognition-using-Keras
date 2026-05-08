"""Emotion predictor with temporal smoothing for stable on-screen labels."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import keras
import numpy as np

from src.config import EMOTION_COLORS, EMOTION_LABELS, InferenceConfig
from src.inference.face_detector import BoundingBox


class EmotionPredictor:
    """Loads a saved Keras model and runs per-face emotion classification.

    A short rolling window of softmax probabilities is averaged before argmax.
    This dramatically reduces label flicker without adding visible latency.
    """

    def __init__(self, model_path: Path, cfg: InferenceConfig | None = None) -> None:
        self.cfg = cfg or InferenceConfig()
        self._model: keras.Model = keras.models.load_model(model_path)
        self._history: dict[int, deque[np.ndarray]] = {}

    @staticmethod
    def _crop(frame_rgb: np.ndarray, box: BoundingBox) -> np.ndarray:
        return frame_rgb[box.y : box.y + box.height, box.x : box.x + box.width]

    def _preprocess(self, faces_rgb: list[np.ndarray]) -> np.ndarray:
        size = self.cfg.image_size
        batch = np.stack(
            [cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA) for face in faces_rgb],
            axis=0,
        ).astype(np.float32)
        return batch  # EfficientNetV2 preprocessing is built into the backbone.

    def predict(
        self,
        frame_rgb: np.ndarray,
        boxes: list[BoundingBox],
    ) -> list[tuple[BoundingBox, str, float]]:
        if not boxes:
            self._history.clear()
            return []

        faces = [self._crop(frame_rgb, b) for b in boxes]
        faces = [f for f in faces if f.size > 0]
        if not faces:
            return []

        batch = self._preprocess(faces)
        probs = self._model.predict(batch, verbose=0)

        results: list[tuple[BoundingBox, str, float]] = []
        for idx, (box, p) in enumerate(zip(boxes, probs, strict=False)):
            window = self._history.setdefault(idx, deque(maxlen=self.cfg.smoothing_window))
            window.append(p)
            smoothed = np.mean(np.stack(window), axis=0)
            class_idx = int(smoothed.argmax())
            results.append((box, EMOTION_LABELS[class_idx], float(smoothed[class_idx])))
        return results

    @staticmethod
    def annotate_frame(
        frame_bgr: np.ndarray,
        results: list[tuple[BoundingBox, str, float]],
    ) -> np.ndarray:
        for box, label, conf in results:
            color = EMOTION_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(
                frame_bgr,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color,
                2,
            )
            text = f"{label} {conf * 100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame_bgr,
                (box.x, box.y - th - baseline - 6),
                (box.x + tw + 6, box.y),
                color,
                -1,
            )
            cv2.putText(
                frame_bgr,
                text,
                (box.x + 3, box.y - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return frame_bgr
