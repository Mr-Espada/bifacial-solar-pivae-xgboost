"""CSV selection, immutable provenance and whole-day cross-fitting."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .config import PAPER_FEATURES, TARGET_Q, TARGET_T, PREDICTED_Q

RAW_HASHES = {
    "train_simulation.csv": "2c63259c638777b0c988ff4919847bc345ceb8489cefb7ac95e54dd73ee2433f",
    "test_simulation.csv": "437dfdeb2c9e02476cdfcaf2a8fafc768be5ab4a3abab099fc3903cde7b7f319",
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_features(features):
    if len(features) != len(set(features)) or not set(features) <= set(PAPER_FEATURES):
        raise ValueError("Only unique paper input variables are permitted")


def read_inputs(path, features, target=None):
    """Inference requests no target column, even when it exists in the raw CSV."""
    check_features(features)
    if target is not None and target not in (TARGET_Q, TARGET_T):
        raise ValueError("Unknown target")
    columns = ["Datum", *features] + ([target] if target else [])
    data = pd.read_csv(path, usecols=columns)
    data = data.loc[:, columns]
    numeric = data.loc[:, list(features) + ([target] if target else [])].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Missing/nonfinite values are not silently imputed")
    dates = pd.to_datetime(data["Datum"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Expected unique chronological rows")
    data.insert(0, "row_id", np.arange(len(data), dtype=int))
    return data


def verify_training_dates(data):
    days = pd.to_datetime(data.Datum).dt.strftime("%Y-%m-%d")
    if (days == "2024-08-28").any():
        raise ValueError("Held-out August 28 must not occur in training")
    allowed = {f"2024-08-{n:02}" for n in range(21, 30)} - {"2024-08-28"}
    if not set(days) <= allowed:
        raise ValueError("Training dates outside the supplied experiment")


def day_folds(data, n_splits):
    groups = pd.to_datetime(data.Datum).dt.strftime("%Y-%m-%d").to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough training days for the requested folds")
    for fold, (fit, predict) in enumerate(GroupKFold(n_splits=n_splits).split(data, groups=groups)):
        fit_days = sorted(set(groups[fit]))
        predict_days = sorted(set(groups[predict]))
        assert not set(fit_days) & set(predict_days)
        yield fold, fit, predict, {
            "fold": fold, "fit_days": fit_days, "prediction_days": predict_days,
            "fit_rows": len(fit), "prediction_rows": len(predict),
        }


def identity_frame(data):
    return data.loc[:, ["row_id", "Datum"]].copy()


def add_predicted_q(data, features, predictions):
    """Exact row/timestamp alignment; no measured Q or delta-T can enter."""
    check_features(features)
    for col in ("row_id", "Datum"):
        if not np.array_equal(data[col].to_numpy(), predictions[col].to_numpy()):
            raise ValueError(f"Predicted Q {col} does not align with inputs")
    q = predictions[PREDICTED_Q].to_numpy(float)
    if not np.isfinite(q).all():
        raise ValueError("Incomplete/nonfinite predicted Q")
    result = data.loc[:, features].copy()
    result[PREDICTED_Q] = q
    assert TARGET_Q not in result and TARGET_T not in result and "dT_1K" not in result
    return result


def verify_raw(data_dir):
    actual = {name: file_hash(Path(data_dir) / name) for name in RAW_HASHES}
    if actual != RAW_HASHES:
        raise ValueError("Raw CSV hashes differ from the inspected originals")
    return actual


def verify_training_source(path):
    """Fitting checks only the training file; no held-out targets are read."""
    if file_hash(path) != RAW_HASHES["train_simulation.csv"]:
        raise ValueError("Training CSV differs from the inspected original")
