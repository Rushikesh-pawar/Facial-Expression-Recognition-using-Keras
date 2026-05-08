"""MediaPipe-based face detection.

MediaPipe replaces the legacy Haar cascade detector that ships with OpenCV:
* significantly more robust to scale, rotation, and lighting;
* GPU-accelerated where supported;
* still pure-Python to install (no model files to vendor).
"""
from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def clamp(self, frame_w: int, frame_h: int) -> "BoundingBox":
        x = max(0, self.x)
        y = max(0, self.y)
        w = max(1, min(self.width, frame_w - x))
        h = max(1, min(self.height, frame_h - y))
        return BoundingBox(x, y, w, h, self.confidence)


class FaceDetector:
    """Thin wrapper around MediaPipe's short-range face detector."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 = within 2m of camera (good for webcams)
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame_rgb: np.ndarray) -> list[BoundingBox]:
        """Detect faces in an RGB frame and return clamped bounding boxes."""
        h, w, _ = frame_rgb.shape
        results = self._detector.process(frame_rgb)
        if not results.detections:
            return []

        boxes: list[BoundingBox] = []
        for det in results.detections:
            rel = det.location_data.relative_bounding_box
            box = BoundingBox(
                x=int(rel.xmin * w),
                y=int(rel.ymin * h),
                width=int(rel.width * w),
                height=int(rel.height * h),
                confidence=float(det.score[0]),
            ).clamp(w, h)
            boxes.append(box)
        return boxes

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
