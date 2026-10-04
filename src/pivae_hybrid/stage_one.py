"""Generate day-held-out training Q and final training-only piVAE posterior."""

from pathlib import Path
import numpy as np
from .config import TARGET_Q, PREDICTED_Q, training_signature
from .data import read_inputs, verify_training_dates, day_folds, identity_frame, file_hash, write_json, verify_training_source
from .artifacts import model_hashes, verify_posterior_quality
from .training import fit_neural
from .posterior import build_stan_model, condition_posterior, predict_posterior


def train_pivae(cfg, run_dir):
    root = Path(run_dir) / "pivae"
    if root.exists() and any(root.iterdir()):
        raise ValueError("piVAE stage already contains artifacts; choose a new run directory to preserve logs")
    features = cfg["features"]
    source = Path(cfg["data_dir"]) / "train_simulation.csv"
    verify_training_source(source)
    train = read_inputs(source, features, target=TARGET_Q)
    verify_training_dates(train)
    x, y = train[features].to_numpy(), train[TARGET_Q].to_numpy()
    oof = identity_frame(train)
    oof["fold"] = -1
    for name in (PREDICTED_Q, "q_ci_low", "q_ci_high", "q_pi_low", "q_pi_high"):
        oof[name] = np.nan
    stan = build_stan_model(run_dir)
    folds = []
    models = {}
    for fold, fit_rows, prediction_rows, manifest in day_folds(train, cfg["crossfit"]["folds"]):
        output = root / f"fold_{fold}"
        seed = cfg["seed"] + 1000 * (fold + 1)
        model, checkpoint = fit_neural(x[fit_rows], y[fit_rows], features, cfg, seed, output)
        posterior = condition_posterior(model, checkpoint, x[fit_rows], y[fit_rows], cfg, seed, output, stan)
        verify_posterior_quality(output, cfg)
        models[output.name] = model_hashes(output)
        predicted = predict_posterior(model, checkpoint, x[prediction_rows], posterior,
                                      cfg["interval_alpha"], cfg["prediction_chunk_size"], seed)
        for col, values in predicted.items():
            oof.loc[prediction_rows, col] = values
        oof.loc[prediction_rows, "fold"] = fold
        manifest["fit_row_ids_sha256"] = __import__("hashlib").sha256(train.row_id.iloc[fit_rows].to_numpy().tobytes()).hexdigest()
        manifest["seed"] = seed
        write_json(output / "fold.json", manifest)
        folds.append(manifest)
    if oof[PREDICTED_Q].isna().any() or (oof.fold < 0).any():
        raise RuntimeError("Cross-fitting did not predict every training row")
    oof.to_csv(root / "q_train_oof.csv", index=False)
    full = root / "full"
    model, checkpoint = fit_neural(x, y, features, cfg, cfg["seed"], full)
    posterior = condition_posterior(model, checkpoint, x, y, cfg, cfg["seed"], full, stan)
    verify_posterior_quality(full, cfg)
    models["full"] = model_hashes(full)
    fitted = identity_frame(train)
    for col, values in predict_posterior(model, checkpoint, x, posterior, cfg["interval_alpha"],
                                         cfg["prediction_chunk_size"], cfg["seed"]).items():
        fitted[col] = values
    fitted.to_csv(root / "q_train_fitted.csv", index=False)
    write_json(root / "manifest.json", {
        "training_signature": training_signature(cfg), "training_sha256": file_hash(source),
        "features": features, "target": TARGET_Q, "training_rows": len(train), "folds": folds,
        "oof_sha256": file_hash(root / "q_train_oof.csv"),
        "fitted_sha256": file_hash(root / "q_train_fitted.csv"), "models": models,
        "oof_role": "Every Q feature is predicted by a complete first stage excluding its entire day",
        "posterior_likelihood": "All fitting rows, exact augmented-QR Gaussian likelihood",
    })
