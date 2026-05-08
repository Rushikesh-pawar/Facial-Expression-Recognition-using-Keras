# Facial Expression Recognition

Real-time, in-browser facial-expression classifier built with **Keras 3 / TensorFlow 2.18**, **EfficientNetV2** transfer learning, and **MediaPipe** face detection. The trained model serves predictions through a Flask app that streams an MJPEG webcam feed with on-frame emotion labels.

> **Stack:** Python 3.11 · Keras 3 · TensorFlow 2.18 · EfficientNetV2-B0 · MediaPipe · OpenCV · Flask · Docker

---

## Highlights

- **Modern transfer-learning recipe.** EfficientNetV2-B0 backbone (ImageNet pretraining) with a custom classifier head, trained in two stages (frozen-head warm-up → top-layer fine-tuning) using **AdamW**, **label smoothing**, **inverse-frequency class weighting**, and **mixed-precision** on GPU.
- **Production-shape inference path.** MediaPipe replaces the legacy Haar-cascade detector for robust face detection; predictions are temporally smoothed across a rolling window to eliminate label flicker.
- **End-to-end pipeline.** Reproducible `tf.data` input pipeline, two-stage trainer with TensorBoard / CSV logging, evaluation script that emits a confusion matrix and per-class report, single-image CLI, and a containerized Flask demo app.
- **Engineering polish.** Type-hinted, modular `src/` layout with a single `TrainingConfig` source of truth; pinned dependencies via `requirements.txt`; `pyproject.toml` with Ruff + pytest; Dockerfile for CPU-only deployment.

---

## Demo

```
┌──────────────────────────┐
│  webcam frame (BGR)      │
└──────────┬───────────────┘
           ▼
   MediaPipe face detector  ──► bounding boxes
           ▼
   crop + resize 224×224
           ▼
   EfficientNetV2-B0  ──► 7-class softmax
           ▼
   rolling-window smoothing
           ▼
   OpenCV overlay → MJPEG → browser
```

Visit <http://localhost:5000> after starting the app.

---

## Project layout

```
.
├── app/
│   └── app.py                  # Flask app + MJPEG streaming endpoint
├── src/
│   ├── config.py               # TrainingConfig / InferenceConfig dataclasses
│   ├── data/fer_dataset.py     # tf.data pipeline + class weighting
│   ├── models/emotion_classifier.py  # EfficientNetV2-B0 transfer model
│   ├── inference/
│   │   ├── face_detector.py    # MediaPipe wrapper
│   │   └── predictor.py        # smoothed emotion prediction + overlay
│   ├── train.py                # two-stage training entrypoint
│   └── evaluate.py             # confusion matrix + classification report
├── scripts/
│   ├── download_fer2013.sh     # Kaggle CLI dataset fetch
│   └── predict_image.py        # single-image CLI
├── templates/index.html        # responsive dark UI
├── static/style.css
├── tests/                      # pytest data-pipeline tests
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) (35,887 grayscale 48×48 face images across 7 emotion classes):

```bash
./scripts/download_fer2013.sh   # requires `pip install kaggle` + ~/.kaggle/kaggle.json
```

This drops `fer2013.csv` into `./data/`.

### 3. Train

```bash
python -m src.train --csv data/fer2013.csv
```

Defaults: 10 head epochs (lr=1e-3, frozen backbone) followed by 25 fine-tune epochs (lr=1e-4, top backbone layers unfrozen). Override per run:

```bash
python -m src.train --csv data/fer2013.csv --epochs-head 5 --epochs-finetune 15 --batch-size 32
```

Artifacts:
- `artifacts/models/emotion_classifier_final.keras`
- `artifacts/logs/{head,finetune}/` — TensorBoard event files + CSV history
- `artifacts/reports/test_metrics.json`

Stream metrics live:

```bash
tensorboard --logdir artifacts/logs
```

### 4. Evaluate

```bash
python -m src.evaluate \
    --model artifacts/models/emotion_classifier_final.keras \
    --csv data/fer2013.csv
```

Writes `artifacts/reports/confusion_matrix.png` and `artifacts/reports/classification_report.json`.

### 5. Run the demo

```bash
python -m app.app --model artifacts/models/emotion_classifier_final.keras
```

Or as a container:

```bash
docker build -t fer-app .
docker run --rm -p 5000:5000 \
    -v "$(pwd)/artifacts:/app/artifacts" \
    --device=/dev/video0 \
    fer-app
```

### 6. One-shot image prediction

```bash
python scripts/predict_image.py \
    --model artifacts/models/emotion_classifier_final.keras \
    --image path/to/photo.jpg \
    --output annotated.jpg
```

---

## Training methodology

| Stage | Backbone | Optimizer | LR | Epochs | Notes |
|------|---------|-----------|-----|--------|------|
| 1 — head warm-up | Frozen EfficientNetV2-B0 | AdamW | 1e-3 | 10 | Trains the new classifier head only |
| 2 — fine-tuning  | Top layers unfrozen (BN frozen) | AdamW | 1e-4 | 25 | End-to-end refinement with `ReduceLROnPlateau` |

Other techniques applied:

- **Label smoothing** (0.1) and **L2 weight decay** (1e-4) for regularization.
- **Inverse-frequency class weighting** to counter FER-2013's heavy "disgust" imbalance (~547 vs. ~7,215 for "happy").
- **Mixed-precision** (`mixed_float16`) on GPU for ~1.5–2× throughput.
- **Augmentation** via Keras preprocessing layers: random horizontal flip, ±8° rotation, ±10% zoom/brightness, ±15% contrast — applied on-GPU as part of the `tf.data` graph.
- **Smoothing window** of 5 softmax frames on inference to stabilize labels at 30 FPS.

---

## Emotion classes

`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

---

## Testing & linting

```bash
pytest          # pure-Python data-pipeline tests
ruff check .    # style + simple correctness
```

---

## License

MIT — see [LICENSE](LICENSE).
