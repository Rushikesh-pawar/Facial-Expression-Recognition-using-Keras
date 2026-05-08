"""Flask app that streams MJPEG video with on-frame emotion annotations.

Run:
    python -m app.app --model artifacts/models/emotion_classifier_final.keras
or, with gunicorn:
    MODEL_PATH=artifacts/models/emotion_classifier_final.keras \
        gunicorn -w 1 -b 0.0.0.0:5000 'app.app:create_app()'
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

import cv2
from flask import Flask, Response, render_template

from src.config import PROJECT_ROOT, InferenceConfig
from src.inference import EmotionPredictor, FaceDetector

log = logging.getLogger(__name__)

# A camera handle is process-global; serializing access keeps OpenCV happy.
_capture_lock = threading.Lock()


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _frame_generator(model_path: Path, cfg: InferenceConfig):
    detector = FaceDetector(min_confidence=cfg.detection_confidence)
    predictor = EmotionPredictor(model_path, cfg)
    capture = _open_camera(cfg.camera_index)
    log.info("camera %d opened; streaming...", cfg.camera_index)
    try:
        while True:
            with _capture_lock:
                ok, frame_bgr = capture.read()
            if not ok:
                log.warning("camera read failed; stopping stream")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes = detector.detect(frame_rgb)
            results = predictor.predict(frame_rgb, boxes)
            annotated = predictor.annotate_frame(frame_bgr, results)

            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
    finally:
        capture.release()
        detector.close()
        log.info("camera released")


def create_app(model_path: Path | None = None, cfg: InferenceConfig | None = None) -> Flask:
    """Application factory; works with both ``flask run`` and gunicorn."""
    cfg = cfg or InferenceConfig()
    if model_path is None:
        env_path = os.environ.get("MODEL_PATH")
        model_path = Path(env_path) if env_path else PROJECT_ROOT / "artifacts" / "models" / "emotion_classifier_final.keras"

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed() -> Response:
        return Response(
            _frame_generator(model_path, cfg),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/healthz")
    def healthz() -> tuple[str, int]:
        return "ok", 200

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FER Flask demo app.")
    parser.add_argument("--model", type=Path, default=None, help="Path to .keras model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    cfg = InferenceConfig(camera_index=args.camera)
    app = create_app(args.model, cfg)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
