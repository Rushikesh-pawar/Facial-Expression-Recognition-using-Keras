"""CLI: run the trained classifier on a single image and print the result."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.config import InferenceConfig
from src.inference import EmotionPredictor, FaceDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict the emotion in a single image.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save annotated image")
    args = parser.parse_args()

    frame_bgr = cv2.imread(str(args.image))
    if frame_bgr is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cfg = InferenceConfig()
    with FaceDetector(min_confidence=cfg.detection_confidence) as detector:
        boxes = detector.detect(frame_rgb)
    if not boxes:
        print("no faces detected")
        return 0

    predictor = EmotionPredictor(args.model, cfg)
    results = predictor.predict(frame_rgb, boxes)
    for i, (_, label, conf) in enumerate(results):
        print(f"face[{i}]: {label} ({conf * 100:.1f}%)")

    if args.output:
        annotated = predictor.annotate_frame(frame_bgr.copy(), results)
        cv2.imwrite(str(args.output), annotated)
        print(f"saved annotated image to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
