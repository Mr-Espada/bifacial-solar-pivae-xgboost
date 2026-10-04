"""Integrity checks for saved artifacts; hashes do not establish ownership."""

import json
import math
from pathlib import Path
from .config import training_signature
from .data import file_hash


def package_hashes():
    package = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(package.iterdir())
            if p.suffix in (".py", ".stan")}


def model_hashes(directory):
    directory = Path(directory)
    return {name: file_hash(directory / name)
            for name in ("neural.pt", "posterior.npz", "diagnostics.json")}


def verify_posterior_quality(directory, cfg):
    path = Path(directory) / "diagnostics.json"
    if not path.exists():
        raise ValueError("Missing posterior quality diagnostics; downstream use refused")
    diagnostics = json.loads(path.read_text())
    try:
        rhat = float(diagnostics["max_rhat"])
        ess = float(diagnostics["min_bulk_ess"])
        bfmi = [float(value) for value in diagnostics["bfmi_per_chain"]]
        finite = all(math.isfinite(value) for value in (rhat, ess, *bfmi))
        tail = diagnostics.get("min_tail_ess")
        if tail is not None:
            finite = finite and math.isfinite(float(tail))
        passed = (
            diagnostics.get("quality_passed") is True and finite and bool(bfmi)
            and diagnostics.get("all_relevant_diagnostics_finite", True) is True
            and rhat <= cfg["posterior"]["rhat_threshold"]
            and ess >= cfg["posterior"]["min_ess"]
            and diagnostics["divergences"] == 0
            and diagnostics["max_treedepth_hits"] == 0
            and all(value > 0.3 for value in bfmi)
        )
    except (KeyError, TypeError, ValueError):
        passed = False
    if not passed:
        raise ValueError("Posterior quality gate failed; downstream use refused")


def verify_xgboost_pivae_binding(run_dir, manifest):
    if "hybrid_t" not in manifest["models"]:
        return
    source = Path(run_dir) / "pivae" / "manifest.json"
    expected = manifest.get("pivae_manifest_sha256")
    if expected is None or not source.exists() or expected != file_hash(source):
        raise ValueError("Hybrid tree belongs to a different or unbound piVAE stage")


def verify_pivae_model(root, manifest, name, cfg):
    if manifest["training_signature"] != training_signature(cfg):
        raise ValueError("Frozen piVAE and configuration differ")
    if manifest.get("features") != cfg["features"]:
        raise ValueError("Frozen piVAE feature order differs")
    expected = manifest.get("models", {}).get(name)
    if expected is None:
        raise ValueError("Unbound legacy piVAE artifacts; use scripts/import_saved_run.py on an audited copy")
    if expected != model_hashes(Path(root) / name):
        raise ValueError(f"piVAE neural/posterior artifacts changed after binding: {name}")
    verify_posterior_quality(Path(root) / name, cfg)


def verify_saved_predictions(cfg, run_dir):
    root = Path(run_dir)
    path = root / "predictions" / "manifest.json"
    if not path.exists():
        raise ValueError("Unbound legacy predictions; use scripts/import_saved_run.py on an audited copy")
    manifest = json.loads(path.read_text())
    if manifest["training_signature"] != training_signature(cfg):
        raise ValueError("Prediction configuration differs")
    if manifest["input_sha256"] != file_hash(Path(cfg["data_dir"]) / "test_simulation.csv"):
        raise ValueError("Predictions belong to a different held-out file")
    if manifest["predictions_sha256"] != file_hash(root / "predictions" / "test.csv"):
        raise ValueError("Saved predictions changed after binding")
    for stage, expected in manifest["stage_manifests"].items():
        if expected != file_hash(root / stage / "manifest.json"):
            raise ValueError("Prediction model provenance changed")


def verify_evaluation_sources(cfg, run_dir):
    root = Path(run_dir)
    train_hash = file_hash(Path(cfg["data_dir"]) / "train_simulation.csv")
    for stage in ("pivae", "xgboost"):
        path = root / stage / "manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        if manifest["training_signature"] != training_signature(cfg) or manifest["training_sha256"] != train_hash:
            raise ValueError("Evaluation training provenance differs")
        if stage == "pivae":
            for name in manifest["models"]:
                verify_pivae_model(root / stage, manifest, name, cfg)
        else:
            verify_xgboost_pivae_binding(root, manifest)
        for name, field in (("q_train_oof.csv", "oof_sha256"),
                            ("q_train_fitted.csv", "fitted_sha256") if stage == "pivae"
                            else ("train_predictions.csv", "train_predictions_sha256")):
            if manifest.get(field) != file_hash(root / stage / name):
                raise ValueError(f"Evaluation source changed or lacks binding: {stage}/{name}")
