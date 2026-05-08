"""EfficientNetV2-B0 transfer-learning classifier for FER-2013.

Why this architecture:
* EfficientNetV2 reaches strong ImageNet accuracy at a fraction of the compute
  of ResNet/Inception families, which matters for real-time webcam inference.
* The B0 variant is small enough to run on CPU at >15 FPS while still
  benefiting from rich ImageNet pretraining.
"""
from __future__ import annotations

import keras
from keras import layers

from src.config import TrainingConfig


def build_emotion_classifier(cfg: TrainingConfig) -> keras.Model:
    """Build the two-stage transfer-learning model.

    The backbone is frozen on construction; call :func:`unfreeze_top_layers`
    before the fine-tuning stage.
    """
    inputs = keras.Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    # EfficientNetV2 expects raw 0-255 inputs; preprocessing is built in.
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    backbone.trainable = False

    x = layers.Dropout(cfg.dropout_rate, name="head_dropout_1")(backbone.output)
    x = layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=keras.regularizers.l2(cfg.weight_decay),
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(cfg.dropout_rate, name="head_dropout_2")(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", dtype="float32", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="emotion_classifier_efficientnetv2b0")
    return model


def unfreeze_top_layers(model: keras.Model, unfreeze_from_layer: int) -> int:
    """Unfreeze the backbone from ``unfreeze_from_layer`` to the end.

    BatchNormalization layers are kept frozen even when "unfrozen" because
    fine-tuning their running statistics on a small dataset usually hurts.

    Returns the number of trainable layers after the operation.
    """
    backbone_layers = [layer for layer in model.layers if layer.name.startswith("efficientnetv2")]
    if not backbone_layers:
        # Backbone is wired in directly via `input_tensor`; iterate the full graph.
        candidate_layers = model.layers
    else:
        candidate_layers = backbone_layers

    for i, layer in enumerate(candidate_layers):
        if i < unfreeze_from_layer:
            layer.trainable = False
            continue
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return sum(1 for layer in model.layers if layer.trainable)
