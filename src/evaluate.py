"""Generate a confusion matrix and per-class report on the test split."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import EMOTION_LABELS, REPORTS_DIR, TrainingConfig
from src.data import build_datasets

log = logging.getLogger(__name__)


def _collect_predictions(model: keras.Model, ds) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_true.append(batch_y.numpy())
        y_pred.append(probs.argmax(axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def _save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    normalized = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={"label": "fraction of true class"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("FER-2013 Confusion Matrix (row-normalized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate(model_path: Path, csv_path: Path) -> dict:
    cfg = TrainingConfig()
    model = keras.models.load_model(model_path)
    _, _, test_ds, _ = build_datasets(csv_path, cfg)

    y_true, y_pred = _collect_predictions(model, test_ds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LABELS))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=list(EMOTION_LABELS),
        output_dict=True,
        zero_division=0,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_png = REPORTS_DIR / "confusion_matrix.png"
    report_json = REPORTS_DIR / "classification_report.json"
    _save_confusion_matrix(cm, cm_png)
    report_json.write_text(json.dumps(report, indent=2))

    log.info("confusion matrix saved to %s", cm_png)
    log.info("classification report saved to %s", report_json)
    log.info("accuracy: %.4f", report["accuracy"])
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained emotion classifier.")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved .keras model")
    parser.add_argument("--csv", type=Path, required=True, help="Path to fer2013.csv")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    evaluate(args.model, args.csv)


if __name__ == "__main__":
    main()
