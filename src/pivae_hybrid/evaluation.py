"""Scoring is the only experiment component that reads held-out targets."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from .config import TARGET_Q, TARGET_T, PREDICTED_Q
from .data import write_json, verify_raw
from .artifacts import verify_saved_predictions, verify_evaluation_sources


def regression_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise ValueError("Expected aligned, nonempty one-dimensional predictions")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics do not silently omit nonfinite observations")
    residual = predicted - truth
    denominator = np.sum((truth - truth.mean())**2)
    variable = len(truth) > 1 and np.std(truth) > 0 and np.std(predicted) > 0
    return {
        "n": len(truth), "r2": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else None,
        "pearson": float(pearsonr(truth, predicted).statistic) if variable else None,
        "spearman": float(spearmanr(truth, predicted).statistic) if variable else None,
        "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def interval_metrics(truth, low, high):
    truth, low, high = (np.asarray(x, float) for x in (truth, low, high))
    if truth.shape != low.shape or low.shape != high.shape or truth.ndim != 1:
        raise ValueError("Interval arrays must align")
    if not all(np.isfinite(x).all() for x in (truth, low, high)) or (low > high).any():
        raise ValueError("Nonfinite or reversed intervals")
    return {"coverage": float(np.mean((truth >= low) & (truth <= high))),
            "mean_width": float(np.mean(high - low)), "median_width": float(np.median(high - low))}


def aligned_truth(path, predictions):
    truth = pd.read_csv(path, usecols=["Datum", TARGET_Q, TARGET_T])
    if len(truth) != len(predictions) or not np.array_equal(truth.Datum, predictions.Datum):
        raise ValueError("Predictions do not align with raw target timestamps")
    if not np.array_equal(predictions.row_id, np.arange(len(truth))):
        raise ValueError("Predictions are not in immutable raw row order")
    return truth


def evaluate(cfg, run_dir):
    root = Path(run_dir)
    hashes = verify_raw(cfg["data_dir"])
    verify_saved_predictions(cfg, root)
    verify_evaluation_sources(cfg, root)
    records = []
    target_map = {"pivae_q": TARGET_Q, "xgb_q": TARGET_Q, "xgb_t": TARGET_T,
                  "hybrid_t": TARGET_T, "xgb_q_t": TARGET_T}

    def add(model, target, split, frame, column, truth, intervals=False):
        row = {"model": model, "target": target, "split": split,
               "unit": "kW" if target == TARGET_Q else "degC", **regression_metrics(truth[target].to_numpy(), frame[column].to_numpy())}
        if intervals:
            for kind in ("ci", "pi"):
                scores = interval_metrics(truth[target], frame[f"q_{kind}_low"], frame[f"q_{kind}_high"])
                row.update({f"{kind}_{key}": value for key, value in scores.items()})
        records.append(row)

    test = pd.read_csv(root / "predictions" / "test.csv")
    truth = aligned_truth(Path(cfg["data_dir"]) / "test_simulation.csv", test)
    for model, target in target_map.items():
        if model in test:
            add(model, target, "test_aug28", test, model, truth, intervals=model == "pivae_q")
    train_path = Path(cfg["data_dir"]) / "train_simulation.csv"
    if (root / "pivae" / "manifest.json").exists():
        for filename, split in (("q_train_fitted.csv", "train_fit"), ("q_train_oof.csv", "train_oof")):
            frame = pd.read_csv(root / "pivae" / filename)
            train_truth = aligned_truth(train_path, frame)
            add("pivae_q", TARGET_Q, split, frame, PREDICTED_Q, train_truth, intervals=True)
    if (root / "xgboost" / "manifest.json").exists():
        frame = pd.read_csv(root / "xgboost" / "train_predictions.csv")
        train_truth = aligned_truth(train_path, frame)
        for model, target in target_map.items():
            if model in frame:
                split = "train_stage2_fit_oof_q" if model in ("hybrid_t", "xgb_q_t") else "train_fit"
                add(model, target, split, frame, model, train_truth)
        add("xgb_q", TARGET_Q, "train_oof", frame, "xgb_q_oof", train_truth)
    pd.DataFrame(records).to_csv(root / "metrics.csv", index=False)
    diagnostics = {}
    for path in sorted((root / "pivae").glob("*/diagnostics.json")):
        diagnostics[path.parent.name] = json.loads(path.read_text())
    write_json(root / "metrics.json", {"metrics": records, "posterior_diagnostics": diagnostics,
                                     "raw_csv_sha256": hashes, "interval_probability": 1-cfg["interval_alpha"]})
    from .plots import make_plots
    make_plots(cfg, root, test)
    write_report(cfg, root, records, diagnostics)
    return records


def write_report(cfg, root, records, diagnostics):
    held = [r for r in records if r["split"] == "test_aug28"]
    label = {"pivae_q": "piVAE Q", "xgb_q": "XGBoost Q", "xgb_t": "XGBoost outlet (direct)",
             "hybrid_t": "piVAE Q -> XGBoost outlet", "xgb_q_t": "XGBoost Q -> XGBoost outlet"}
    def fmt(value):
        return "undefined" if value is None else f"{value:.6f}"
    lines = ["# Reconstructed experiment results", "", f"Profile: `{cfg['profile']}`. Seed: `{cfg['seed']}`.", "",
             "These are measured reconstruction results, not a claim to replicate the paper's almost-perfect numbers. The paper omits the final settings and source fitted models are absent.", "",
             "Training uses the original 130,048 rows on eight dates. Scoring uses all 17,280 observations on the held-out August 28, 2024. Training includes August 29 as in the paper; this is held-out-day simulation, not future forecasting.", "",
             "## Held-out metrics", "", "| Model | Target unit | R2 | Pearson | Spearman | MAE | RMSE |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in held:
        lines.append(f"| {label[row['model']]} | {row['unit']} | " + " | ".join(fmt(row[k]) for k in ("r2", "pearson", "spearman", "mae", "rmse")) + " |")
    by_name = {r["model"]: r for r in held}
    if "hybrid_t" in by_name:
        direct, hybrid = by_name["xgb_t"], by_name["hybrid_t"]
        lines += ["", f"Hybrid minus direct outlet MAE: {hybrid['mae']-direct['mae']:+.6f} degC; RMSE: {hybrid['rmse']-direct['rmse']:+.6f} degC. Negative values favor the hybrid.",
                  "", "Q-model and outlet-model errors have different units; compare piVAE with XGBoost Q, and compare the three outlet models with each other."]
    if "pivae_q" in by_name:
        q = by_name["pivae_q"]
        lines += ["", "## Posterior intervals on held-out measured Q", "", "| 95% interval | Empirical coverage | Mean width (kW) | Median width (kW) |", "|---|---:|---:|---:|"]
        for kind, name in (("ci", "Latent function credible"), ("pi", "Observation posterior predictive")):
            lines.append(f"| {name} | {100*q[kind+'_coverage']:.3f}% | {q[kind+'_mean_width']:.6f} | {q[kind+'_median_width']:.6f} |")
        lines += ["", "The latent credible interval conditions on the learned feature map/decoder. Comparing it with noisy measured Q does not establish nominal observation coverage. Posterior predictive intervals also include the Stan observation noise. Neither interval includes neural-weight uncertainty or accounts fully for temporal dependence."]
    opts, hmc = cfg["pivae"], cfg["posterior"]
    lines += ["", "## Settings and diagnostics", "",
              f"Neural fits: {opts['epochs']} fixed epochs, {opts['n_centers']} RBF centers, feature hidden widths {opts['feature_hidden']}, beta dimension {opts['beta_dim']}, latent dimension {opts['latent_dim']}, VAE widths {opts['vae_hidden']}, {opts['coefficient_replicas']} persistent coefficient replicas, batch size {opts['batch_size']}, learning rate {opts['learning_rate']}, KL weight {opts['kl_weight']}.",
              "", f"Stack training: {cfg['crossfit']['folds']} whole-day folds; each scaler, center set, feature map, VAE and HMC likelihood excludes the predicted training day's rows. Final test Q uses a separate fit on all training rows.",
              "", f"Each HMC fit: {hmc['chains']} chains, {hmc['warmup']} warmup and {hmc['samples']} retained draws per chain. Gaussian likelihood uses all fitting observations via exact QR statistics; no observation subsampling.",
              "", f"HMC initialization: {hmc.get('initialization', 'encoder')}; metric: {hmc.get('metric', 'diag_e')}. Multi-start initialization uses only the fitting likelihood and is an optimization aid, not the point predictor. All prediction/interval samples still come from HMC. `initialization.json` records optimization attempts and the selected density; attempts need not represent distinct modes. Convergence gates describe exploration of the initialized mode and do not certify global exploration of every mode.",
              "", "| Posterior | Max Rhat | Min bulk ESS | Divergences | Depth hits | Quality gate |", "|---|---:|---:|---:|---:|---|"]
    for name, diag in diagnostics.items():
        lines.append(f"| {name} | {diag['max_rhat']:.4f} | {diag['min_bulk_ess']:.1f} | {diag['divergences']} | {diag['max_treedepth_hits']} | {'pass' if diag['quality_passed'] else 'FAIL'} |")
    failed = [name for name, diag in diagnostics.items() if not diag["quality_passed"]]
    if failed:
        lines += ["", f"**Posterior quality is incomplete for {', '.join(failed)}. Its point estimates, credible intervals and stack features are provisional. Rerun with more warmup/samples or revise training-only settings before drawing scientific conclusions.**"]
    lines += ["", "Quality gate requires Rhat at or below the configured threshold, minimum bulk ESS, no divergences/depth saturation and BFMI above 0.3 in each chain. Exact diagnostics and CmdStan reports are saved per fit.",
              "", "## Reproduction artifacts", "",
              "`resolved_config.json`, `environment.json`, stage manifests, original CSV hashes, fitted neural/tree models, posterior samples and Stan chain CSVs identify this run. `metrics.json` and `metrics.csv` include separate training-fit and first-stage out-of-fold scores. Outlet training scores fit stage two using out-of-fold Q; they are not out-of-fold outlet scores.",
              "", "Run `python -m pivae_hybrid.cli evaluate --config <same config> --run-dir <this run>` to regenerate every metric and plot without fitting. See the project README for individual stage commands and `docs/RECONSTRUCTION_SOURCE_AUDIT.md` for paper/source discrepancies.",
              "", "Plots show all held-out rows; scatter and combined-record lines use deterministic display thinning only. No metrics are thinned. XGBoost importance is training gain, not a causal explanation."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
