"""Comparable XGBoost baselines and stacks trained with out-of-fold Q."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .config import TARGET_Q, TARGET_T, PREDICTED_Q, training_signature
from .data import read_inputs, verify_training_dates, day_folds, identity_frame, add_predicted_q, file_hash, write_json, verify_training_source
from .artifacts import verify_pivae_model

MODEL_TARGETS = {"xgb_q": TARGET_Q, "xgb_t": TARGET_T, "hybrid_t": TARGET_T, "xgb_q_t": TARGET_T}


def regressor(cfg, seed):
    return XGBRegressor(**cfg["xgboost"], random_state=seed)


def validate_pivae_oof(cfg, run_dir, train):
    root = Path(run_dir) / "pivae"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["training_signature"] != training_signature(cfg):
        raise ValueError("piVAE configuration does not match this stage")
    if manifest["training_sha256"] != file_hash(Path(cfg["data_dir"]) / "train_simulation.csv"):
        raise ValueError("piVAE was fitted on a different training file")
    expected_names = {"full", *(f"fold_{fold}" for fold in range(cfg["crossfit"]["folds"]))}
    if set(manifest.get("models", {})) != expected_names:
        raise ValueError("piVAE model bindings do not cover every whole-day fold and full fit")
    for name in sorted(expected_names):
        verify_pivae_model(root, manifest, name, cfg)
    path = root / "q_train_oof.csv"
    if manifest["oof_sha256"] != file_hash(path):
        raise ValueError("piVAE out-of-fold predictions changed after fitting")
    oof = pd.read_csv(path)
    expected = np.full(len(train), -1, dtype=int)
    for fold, _, held, _ in day_folds(train, cfg["crossfit"]["folds"]):
        expected[held] = fold
    if not np.array_equal(expected, oof["fold"].to_numpy()):
        raise ValueError("piVAE Q features do not have the required whole-day fold assignments")
    # Check ordering and finite predicted-Q values before accepting this artifact.
    add_predicted_q(train, cfg["features"], oof)
    return oof


def train_xgboost(cfg, run_dir, baseline_only=False):
    root = Path(run_dir) / "xgboost"
    if root.exists() and any(root.iterdir()):
        raise ValueError("XGBoost stage already contains artifacts; choose a new run directory to preserve logs")
    features = cfg["features"]
    source = Path(cfg["data_dir"]) / "train_simulation.csv"
    verify_training_source(source)
    q_data = read_inputs(source, features, target=TARGET_Q)
    t_data = read_inputs(source, features, target=TARGET_T)
    verify_training_dates(q_data)
    train = identity_frame(q_data)
    pivae_oof = None if baseline_only else validate_pivae_oof(cfg, run_dir, t_data)
    root.mkdir(parents=True, exist_ok=True)
    x, yq = q_data.loc[:, features], q_data[TARGET_Q].to_numpy()
    oof = identity_frame(q_data)
    oof[PREDICTED_Q] = np.nan
    oof["fold"] = -1
    folds = []
    for fold, fit_rows, prediction_rows, manifest in day_folds(q_data, cfg["crossfit"]["folds"]):
        model = regressor(cfg, cfg["seed"] + 1000 * (fold + 1))
        model.fit(x.iloc[fit_rows], yq[fit_rows])
        oof.loc[prediction_rows, PREDICTED_Q] = model.predict(x.iloc[prediction_rows])
        oof.loc[prediction_rows, "fold"] = fold
        model.save_model(root / f"q_fold_{fold}.json")
        folds.append(manifest)
    oof.to_csv(root / "q_train_oof.csv", index=False)
    matrices = {
        "xgb_q": x, "xgb_t": x,
        "xgb_q_t": add_predicted_q(t_data, features, oof),
    }
    if not baseline_only:
        matrices["hybrid_t"] = add_predicted_q(t_data, features, pivae_oof)
    metadata = {}
    for name, inputs in matrices.items():
        target = yq if name == "xgb_q" else t_data[TARGET_T].to_numpy()
        model = regressor(cfg, cfg["seed"])
        model.fit(inputs, target)
        model.save_model(root / f"{name}.json")
        train[name] = model.predict(inputs)
        importance = model.get_booster().get_score(importance_type="gain")
        pd.DataFrame({"feature": list(inputs.columns), "gain": [importance.get(f, 0.0) for f in inputs.columns]}).to_csv(root / f"{name}_importance.csv", index=False)
        metadata[name] = {"target": MODEL_TARGETS[name], "features": list(inputs.columns),
                          "q_source": {"hybrid_t": "pivae_whole_day_oof", "xgb_q_t": "xgboost_whole_day_oof"}.get(name),
                          "model_sha256": file_hash(root / f"{name}.json")}
        print(f"XGBoost fitted: {name}, features={len(inputs.columns)}, rows={len(inputs)}", flush=True)
    train["xgb_q_oof"] = oof[PREDICTED_Q].to_numpy()
    train.to_csv(root / "train_predictions.csv", index=False)
    write_json(root / "manifest.json", {
        "training_signature": training_signature(cfg), "training_sha256": file_hash(source),
        "models": metadata, "folds": folds, "oof_sha256": file_hash(root / "q_train_oof.csv"),
        "train_predictions_sha256": file_hash(root / "train_predictions.csv"),
        "pivae_manifest_sha256": None if baseline_only else file_hash(Path(run_dir) / "pivae" / "manifest.json"),
        "stack_training": "Whole-day OOF Q only; no measured Q in either temperature feature matrix",
        "selection": "Fixed parameters; no eval_set, test targets or test-driven early stopping",
    })
