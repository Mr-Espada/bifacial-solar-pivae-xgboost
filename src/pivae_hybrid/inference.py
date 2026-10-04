"""Frozen-model inference can run on CSV files containing no target columns."""

import json
from pathlib import Path
import numpy as np
from xgboost import XGBRegressor
from .config import training_signature
from .data import read_inputs, identity_frame, add_predicted_q, file_hash, write_json
from .artifacts import verify_pivae_model, verify_xgboost_pivae_binding
from .training import load_neural
from .posterior import predict_posterior


def predict_csv(cfg, run_dir, input_csv, include_pivae=True):
    root = Path(run_dir)
    data = read_inputs(input_csv, cfg["features"])
    output = identity_frame(data)
    pivae_q = None
    if include_pivae:
        manifest = json.loads((root / "pivae" / "manifest.json").read_text())
        verify_pivae_model(root / "pivae", manifest, "full", cfg)
        model, checkpoint = load_neural(root / "pivae" / "full" / "neural.pt")
        if checkpoint["features"] != cfg["features"]:
            raise ValueError("Neural checkpoint feature order differs")
        with np.load(root / "pivae" / "full" / "posterior.npz") as posterior:
            values = predict_posterior(model, checkpoint, data[cfg["features"]].to_numpy(), posterior,
                                       cfg["interval_alpha"], cfg["prediction_chunk_size"], cfg["seed"])
        pivae_q = identity_frame(data)
        for name, predicted in values.items():
            pivae_q[name] = predicted
            output["pivae_q" if name == "predicted_Q_1kW" else name] = predicted
    xgb = root / "xgboost"
    if (xgb / "manifest.json").exists():
        manifest = json.loads((xgb / "manifest.json").read_text())
        if manifest["training_signature"] != training_signature(cfg):
            raise ValueError("Frozen XGBoost and configuration differ")
        verify_xgboost_pivae_binding(root, manifest)
        models = {}
        for name, metadata in manifest["models"].items():
            path = xgb / f"{name}.json"
            if metadata["model_sha256"] != file_hash(path):
                raise ValueError("XGBoost model changed after fitting")
            model = XGBRegressor()
            model.load_model(path)
            models[name] = model
        original = data.loc[:, cfg["features"]]
        for name in ("xgb_q", "xgb_t"):
            output[name] = models[name].predict(original)
        if "hybrid_t" in models:
            if pivae_q is None:
                raise ValueError("Hybrid inference requires the frozen piVAE")
            output["hybrid_t"] = models["hybrid_t"].predict(add_predicted_q(data, cfg["features"], pivae_q))
        if "xgb_q_t" in models:
            q = identity_frame(data)
            q["predicted_Q_1kW"] = output["xgb_q"].to_numpy()
            output["xgb_q_t"] = models["xgb_q_t"].predict(add_predicted_q(data, cfg["features"], q))
    return output


def predict_heldout(cfg, run_dir, baseline_only=False):
    source = Path(cfg["data_dir"]) / "test_simulation.csv"
    predicted = predict_csv(cfg, run_dir, source, include_pivae=not baseline_only)
    days = __import__("pandas").to_datetime(predicted.Datum).dt.strftime("%Y-%m-%d")
    if set(days) != {"2024-08-28"} or len(predicted) != 17280:
        raise ValueError("Expected the complete held-out August 28 test day")
    directory = Path(run_dir) / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(directory / "test.csv", index=False)
    write_json(directory / "manifest.json", {
        "training_signature": training_signature(cfg), "input_sha256": file_hash(source),
        "predictions_sha256": file_hash(directory / "test.csv"),
        "stage_manifests": {stage: file_hash(Path(run_dir) / stage / "manifest.json")
                            for stage in ("pivae", "xgboost")
                            if (Path(run_dir) / stage / "manifest.json").exists()},
    })
    return predicted
