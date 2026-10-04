"""Copy/bind an audited 0.1.0 run; do not retrain or edit the source run."""
import argparse
import io
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from pivae_hybrid.artifacts import model_hashes, package_hashes, verify_posterior_quality
from pivae_hybrid.config import training_signature
from pivae_hybrid.data import file_hash, verify_raw, write_json
from pivae_hybrid.inference import predict_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "source-package", "data-dir", "output-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.output_run.resolve()
    if output.exists() or source == output or source in output.parents:
        parser.error("Choose a new output outside the source run")
    original = json.loads((source / "code_manifest.json").read_text())["files"]
    for name, digest in original.items():
        if digest != file_hash(args.source_package / "src" / "pivae_hybrid" / name):
            raise ValueError("Original package does not match saved run source manifest")
    current = package_hashes()
    for name in ("model.py", "training.py", "config.py", "pivae_2d.stan"):
        if original[name] != current[name]:
            raise ValueError(f"Scientific implementation differs: {name}; compatibility migration refused")
    raw_hashes = verify_raw(args.data_dir)
    cfg = json.loads((source / "resolved_config.json").read_text())
    cfg["data_dir"] = str(args.data_dir.resolve())
    signature = training_signature(cfg)
    for stage in ("pivae", "xgboost"):
        m = json.loads((source / stage / "manifest.json").read_text())
        if m["training_signature"] != signature or m["training_sha256"] != raw_hashes["train_simulation.csv"]:
            raise ValueError("Saved stage configuration/data differs")
        if m["oof_sha256"] != file_hash(source / stage / "q_train_oof.csv"):
            raise ValueError("Saved out-of-fold predictions changed")
        if stage == "xgboost":
            for name, item in m["models"].items():
                if item["model_sha256"] != file_hash(source / stage / f"{name}.json"):
                    raise ValueError("Saved tree model changed")
    before = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*")) if p.is_file()}
    for directory in sorted((source / "pivae").iterdir()):
        if directory.is_dir() and (directory / "neural.pt").exists():
            verify_posterior_quality(directory, cfg)
    shutil.copytree(source, output)
    write_json(output / "resolved_config.json", cfg)
    pi = json.loads((output / "pivae" / "manifest.json").read_text())
    pi["models"] = {p.name: model_hashes(p) for p in sorted((output / "pivae").iterdir())
                    if p.is_dir() and (p / "neural.pt").exists()}
    pi["fitted_sha256"] = file_hash(output / "pivae" / "q_train_fitted.csv")
    write_json(output / "pivae" / "manifest.json", pi)
    tree = json.loads((output / "xgboost" / "manifest.json").read_text())
    tree["train_predictions_sha256"] = file_hash(output / "xgboost" / "train_predictions.csv")
    tree["pivae_manifest_sha256"] = file_hash(output / "pivae" / "manifest.json")
    write_json(output / "xgboost" / "manifest.json", tree)
    saved = pd.read_csv(output / "predictions" / "test.csv")
    fresh = predict_csv(cfg, output, args.data_dir / "test_simulation.csv")
    serialized = pd.read_csv(io.StringIO(fresh.to_csv(index=False)))
    if list(saved) != list(serialized) or not all(np.array_equal(saved[c], serialized[c]) for c in saved):
        raise ValueError("Migrated inference differs; copied run is not validated")
    write_json(output / "predictions" / "manifest.json", {
        "training_signature": signature, "input_sha256": raw_hashes["test_simulation.csv"],
        "predictions_sha256": file_hash(output / "predictions" / "test.csv"),
        "stage_manifests": {s: file_hash(output / s / "manifest.json") for s in ("pivae", "xgboost")}})
    after = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*")) if p.is_file()}
    if before != after:
        raise RuntimeError("Source run changed during import")
    write_json(output / "import_record.json", {
        "operation": "2026 maintenance import, no fitting", "binding_is_post_hoc": True,
        "source_run_files": before, "original_package": original, "maintenance_package": current,
        "full_test_rows": len(saved), "serialized_predictions_unchanged": True,
        "source_run_unchanged": True,
        "limit": "Today's hashes prevent subsequent accidental mixing; no historical authentication, ownership proof or recovery of lost optimizer logs."})
    print(f"Imported {len(saved)} verified prediction rows into {output}; source unchanged")


if __name__ == "__main__":
    main()
