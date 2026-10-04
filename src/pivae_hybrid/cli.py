"""Explicit commands for fitting, inference, evaluation and leakage checks."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from .config import load_config, training_signature
from .data import write_json, verify_raw, file_hash
from .artifacts import package_hashes


def initialize_run(cfg, run_dir):
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = root / "resolved_config.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if training_signature(old) != training_signature(cfg):
            raise ValueError("Run directory belongs to different experiment settings; use a new directory")
    else:
        write_json(existing, cfg)
        versions = {}
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost-cpu", "cmdstanpy", "matplotlib", "pytest"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "not installed"
        import cmdstanpy
        try:
            stan = {"path": cmdstanpy.cmdstan_path(), "version": list(cmdstanpy.cmdstan_version())}
        except ValueError:
            stan = {"path": None, "version": None}
        write_json(root / "environment.json", {"python": sys.version, "platform": platform.platform(),
                                               "packages": versions, "cmdstan": stan,
                                               "training_signature": training_signature(cfg)})
        write_json(root / "code_manifest.json", {
            "files": package_hashes(),
            "purpose": "Source hashes at run initialization; original source inventory is in docs",
        })
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description="piVAE/Stan -> XGBoost experiment without target leakage")
    parser.add_argument("command", choices=["audit", "train-pivae", "train-xgb", "predict", "evaluate", "verify-run", "run-all"])
    parser.add_argument("--config", default="configs/reconstruction.json")
    parser.add_argument("--data-dir", help="Directory with the two original CSV files")
    parser.add_argument("--run-dir", default="runs/reconstruction")
    parser.add_argument("--baseline-only", action="store_true", help="Run tree baselines without piVAE")
    parser.add_argument("--input-csv", help="For predict: arbitrary target-free CSV with original inputs and Datum")
    parser.add_argument("--output-csv", help="For predict with --input-csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    if args.command == "audit":
        print(json.dumps(verify_raw(cfg["data_dir"]), indent=2))
        return
    if args.command in ("train-pivae", "train-xgb", "run-all"):
        manifest_path = Path(args.run_dir) / "code_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["files"] != package_hashes():
            raise ValueError("Source changed since run initialization; use a new run directory for fitting")
    root = initialize_run(cfg, args.run_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib_cache"))
    if args.command == "run-all":
        verify_raw(cfg["data_dir"])
    if args.command in ("train-pivae", "run-all") and not args.baseline_only:
        from .stage_one import train_pivae
        train_pivae(cfg, root)
    if args.command in ("train-xgb", "run-all"):
        from .stage_two import train_xgboost
        train_xgboost(cfg, root, baseline_only=args.baseline_only)
    if args.command in ("predict", "run-all"):
        from .inference import predict_csv, predict_heldout
        if args.input_csv:
            if not args.output_csv:
                parser.error("--output-csv is required with --input-csv")
            result = predict_csv(cfg, root, args.input_csv, include_pivae=not args.baseline_only)
            output = Path(args.output_csv)
            if output.resolve() in {(Path(cfg["data_dir"]) / name).resolve() for name in ("train_simulation.csv", "test_simulation.csv")} or output.resolve() == Path(args.input_csv).resolve():
                raise ValueError("Inference output must not overwrite raw or input data")
            output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False)
        else:
            predict_heldout(cfg, root, baseline_only=args.baseline_only)
    if args.command in ("evaluate", "run-all"):
        from .evaluation import evaluate
        evaluate(cfg, root)
    if args.command in ("verify-run", "run-all"):
        from .verification import verify_inference_invariance
        verify_inference_invariance(cfg, root, baseline_only=args.baseline_only)
    print(f"Completed {args.command}. Artifacts: {root}", flush=True)


if __name__ == "__main__":
    main()
