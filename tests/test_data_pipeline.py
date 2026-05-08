"""Lightweight tests for pure-Python data utilities (no GPU / no model)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src.data.fer_dataset import compute_class_weights, load_fer2013_csv


def _make_fake_csv() -> io.StringIO:
    pixels = " ".join(["0"] * (48 * 48))
    rows = [
        {"emotion": 0, "pixels": pixels, "Usage": "Training"},
        {"emotion": 3, "pixels": pixels, "Usage": "Training"},
        {"emotion": 3, "pixels": pixels, "Usage": "PublicTest"},
        {"emotion": 6, "pixels": pixels, "Usage": "PrivateTest"},
    ]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_load_fer2013_csv_validates_schema(tmp_path) -> None:
    csv_path = tmp_path / "fer2013.csv"
    csv_path.write_text(_make_fake_csv().read())
    df = load_fer2013_csv(csv_path)
    assert {"emotion", "pixels", "Usage"} <= set(df.columns)
    assert len(df) == 4


def test_class_weights_sum_to_num_classes() -> None:
    labels = np.array([0, 0, 0, 3, 3, 6], dtype=np.int32)
    weights = compute_class_weights(labels)
    # Weights are normalized so that sum(count_i * w_i) == total_count.
    total = sum(weights[i] * (labels == i).sum() for i in weights)
    assert np.isclose(total, len(labels))
    # Underrepresented class (6) should weigh more than overrepresented (0).
    assert weights[6] > weights[0]
