"""Two-stage training pipeline: classifier head -> end-to-end fine-tuning."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import keras
import tensorflow as tf

from src.config import (
    LOGS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TrainingConfig,
)
from src.data import build_datasets, compute_class_weights
from src.models import build_emotion_classifier, unfreeze_top_layers

log = logging.getLogger(__name__)


def _enable_mixed_precision() -> None:
    keras.mixed_precision.set_global_policy("mixed_float16")


def _make_callbacks(stage: str, cfg: TrainingConfig) -> list[keras.callbacks.Callback]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = MODELS_DIR / f"emotion_classifier_{stage}.keras"
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / stage),
            histogram_freq=1,
            update_freq="epoch",
        ),
        keras.callbacks.CSVLogger(str(LOGS_DIR / f"{stage}.csv")),
    ]


def _compile(model: keras.Model, lr: float, cfg: TrainingConfig) -> None:
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )


def train(csv_path: Path, cfg: TrainingConfig) -> dict[str, float]:
    """Train the model end-to-end and return final test metrics."""
    keras.utils.set_random_seed(cfg.seed)
    if cfg.mixed_precision and tf.config.list_physical_devices("GPU"):
        _enable_mixed_precision()
        log.info("mixed_float16 precision enabled")

    train_ds, val_ds, test_ds, train_labels = build_datasets(csv_path, cfg)
    class_weight = compute_class_weights(train_labels) if cfg.use_class_weights else None
    if class_weight:
        log.info("class weights: %s", class_weight)

    model = build_emotion_classifier(cfg)
    model.summary(print_fn=log.info)

    # ---- Stage 1: train classifier head ----
    log.info("Stage 1/2: training classifier head (backbone frozen)")
    _compile(model, cfg.head_lr, cfg)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.head_epochs,
        callbacks=_make_callbacks("head", cfg),
        class_weight=class_weight,
        verbose=2,
    )

    # ---- Stage 2: fine-tune top backbone layers ----
    log.info("Stage 2/2: fine-tuning from layer %d", cfg.finetune_unfreeze_from_layer)
    n_trainable = unfreeze_top_layers(model, cfg.finetune_unfreeze_from_layer)
    log.info("trainable layers after unfreeze: %d", n_trainable)
    _compile(model, cfg.finetune_lr, cfg)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.finetune_epochs,
        callbacks=_make_callbacks("finetune", cfg),
        class_weight=class_weight,
        verbose=2,
    )

    # ---- Evaluation ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = MODELS_DIR / "emotion_classifier_final.keras"
    model.save(final_path)
    log.info("saved final model to %s", final_path)

    metrics = model.evaluate(test_ds, return_dict=True, verbose=2)
    metrics_path = REPORTS_DIR / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("test metrics: %s", metrics)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FER-2013 emotion classifier.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to fer2013.csv")
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    cfg = TrainingConfig()
    overrides: dict = {}
    if args.epochs_head is not None:
        overrides["head_epochs"] = args.epochs_head
    if args.epochs_finetune is not None:
        overrides["finetune_epochs"] = args.epochs_finetune
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if overrides:
        cfg = TrainingConfig(**{**cfg.__dict__, **overrides})

    train(args.csv, cfg)


if __name__ == "__main__":
    main()
