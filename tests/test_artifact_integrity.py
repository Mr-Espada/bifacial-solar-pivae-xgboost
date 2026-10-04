from pathlib import Path
import pytest
from pivae_hybrid.artifacts import (model_hashes, verify_pivae_model, verify_saved_predictions,
                                    verify_posterior_quality, verify_xgboost_pivae_binding)
from pivae_hybrid.config import load_config, training_signature
from pivae_hybrid.data import file_hash, write_json
from pivae_hybrid.cli import main
from test_leakage_and_numerics import cfg, synthetic_data


def config():
    return load_config(Path(__file__).parents[1] / "configs" / "verification.json")


def good_diagnostics():
    return {"quality_passed": True, "max_rhat": 1.0, "min_bulk_ess": 800.0,
            "bfmi_per_chain": [1.0, 1.0], "divergences": 0, "max_treedepth_hits": 0}


def test_swapping_same_shape_posterior_is_rejected(tmp_path):
    cfg = config()
    full = tmp_path / "full"
    full.mkdir()
    (full / "neural.pt").write_bytes(b"frozen neural and scaling")
    (full / "posterior.npz").write_bytes(b"posterior from this fit")
    write_json(full / "diagnostics.json", good_diagnostics())
    manifest = {"training_signature": training_signature(cfg), "features": cfg["features"],
                "models": {"full": model_hashes(full)}}
    verify_pivae_model(tmp_path, manifest, "full", cfg)
    (full / "posterior.npz").write_bytes(b"other same-shaped posterior")
    with pytest.raises(ValueError, match="changed after binding"):
        verify_pivae_model(tmp_path, manifest, "full", cfg)


def test_unbound_historical_pair_cannot_pass_silently(tmp_path):
    cfg = config()
    manifest = {"training_signature": training_signature(cfg), "features": cfg["features"]}
    with pytest.raises(ValueError, match="Unbound legacy"):
        verify_pivae_model(tmp_path, manifest, "full", cfg)


def test_metric_input_and_model_provenance_are_bound(tmp_path):
    cfg = config()
    cfg["data_dir"] = str(tmp_path)
    raw = tmp_path / "test_simulation.csv"
    raw.write_text("original test inputs")
    (tmp_path / "predictions").mkdir()
    (tmp_path / "pivae").mkdir()
    pred = tmp_path / "predictions" / "test.csv"
    pred.write_text("original predictions")
    stage = tmp_path / "pivae" / "manifest.json"
    stage.write_text("original fit binding")
    write_json(tmp_path / "predictions" / "manifest.json", {
        "training_signature": training_signature(cfg), "input_sha256": file_hash(raw),
        "predictions_sha256": file_hash(pred), "stage_manifests": {"pivae": file_hash(stage)}})
    verify_saved_predictions(cfg, tmp_path)
    stage.write_text("other fit binding")
    with pytest.raises(ValueError, match="model provenance"):
        verify_saved_predictions(cfg, tmp_path)
    stage.write_text("original fit binding")
    pred.write_text("changed predictions")
    with pytest.raises(ValueError, match="predictions changed"):
        verify_saved_predictions(cfg, tmp_path)


def test_fitting_continuation_refuses_changed_source_before_writing(tmp_path):
    write_json(tmp_path / "code_manifest.json", {"files": {"model.py": "different"}})
    cfg_path = Path(__file__).parents[1] / "configs" / "verification.json"
    with pytest.raises(ValueError, match="Source changed"):
        main(["train-pivae", "--config", str(cfg_path), "--run-dir", str(tmp_path)])
    assert not (tmp_path / "resolved_config.json").exists()


@pytest.mark.parametrize("quality_ok", [True, False])
def test_optimizer_raw_outputs_have_one_directory_per_start(cfg, tmp_path, quality_ok):
    import json
    import numpy as np
    import pandas as pd
    from pivae_hybrid.training import fit_neural
    from pivae_hybrid.posterior import condition_posterior
    raw = synthetic_data()
    x, y = raw[cfg["features"]].to_numpy(), raw.Q_1kW.to_numpy()
    model, checkpoint = fit_neural(x, y, cfg["features"], cfg, 7, tmp_path)
    cfg["posterior"].update(map_starts=3, initialization="multistart_map")
    if not quality_ok:
        cfg["posterior"]["min_ess"] = 900
    directories = []

    class Mode:
        optimized_params_dict = {"lp__": 0.0}
        def stan_variable(self, name):
            return np.zeros(cfg["pivae"]["latent_dim"]) if name == "z" else .025

    class Fit:
        def stan_variable(self, name):
            if name == "sigma":
                return np.full(8, .025)
            size = cfg["pivae"]["beta_dim"] if name == "f" else cfg["pivae"]["latent_dim"]
            return np.zeros((8, size))
        def summary(self):
            return pd.DataFrame({"R_hat": [1.0], "ESS_bulk": [800.0], "ESS_tail": [800.0]}, index=["sigma"])
        def method_variables(self):
            return {"energy__": np.random.default_rng(9).normal(size=(8, 2)),
                    "divergent__": np.zeros((8, 2)), "treedepth__": np.ones((8, 2))}
        def diagnose(self):
            return "Mock sampler: this test verifies log preservation, not statistical inference."

    class Stan:
        def optimize(self, **kwargs):
            directory = Path(kwargs["output_dir"])
            directories.append(directory)
            # Equal timestamp-derived basenames can no longer overwrite another start.
            (directory / "same-second.csv").write_text(str(kwargs["seed"]))
            return Mode()
        def sample(self, **kwargs):
            return Fit()

    if quality_ok:
        condition_posterior(model, checkpoint, x, y, cfg, 7, tmp_path, Stan())
    else:
        with pytest.raises(RuntimeError, match="Posterior quality gate failed"):
            condition_posterior(model, checkpoint, x, y, cfg, 7, tmp_path, Stan())
    assert json.loads((tmp_path / "diagnostics.json").read_text())["quality_passed"] is quality_ok
    assert len(set(directories)) == 3
    assert [(p / "same-second.csv").read_text() for p in directories] == ["7", "8", "9"]
    attempts = json.loads((tmp_path / "initialization.json").read_text())["modes"]
    assert all("same-second.csv" in attempt["outputs"] for attempt in attempts)


@pytest.mark.parametrize("change", [
    {"quality_passed": False}, {"max_rhat": 1.5}, {"min_bulk_ess": 1.0},
    {"bfmi_per_chain": ["NaN"]}, {"divergences": 1}, {"max_treedepth_hits": 1},
])
def test_posterior_gate_rechecks_diagnostics_instead_of_trusting_flag(tmp_path, change):
    diagnostics = good_diagnostics()
    diagnostics.update(change)
    write_json(tmp_path / "diagnostics.json", diagnostics)
    with pytest.raises(ValueError, match="quality gate failed"):
        verify_posterior_quality(tmp_path, config())


def test_missing_quality_report_is_rejected_and_valid_report_is_hash_bound(tmp_path):
    with pytest.raises(ValueError, match="Missing posterior"):
        verify_posterior_quality(tmp_path, config())
    full = tmp_path / "full"
    full.mkdir()
    (full / "neural.pt").write_bytes(b"neural")
    (full / "posterior.npz").write_bytes(b"posterior")
    write_json(full / "diagnostics.json", good_diagnostics())
    cfg = config()
    manifest = {"training_signature": training_signature(cfg), "features": cfg["features"],
                "models": {"full": model_hashes(full)}}
    verify_pivae_model(tmp_path, manifest, "full", cfg)
    changed = good_diagnostics()
    changed["max_rhat"] = 1.01
    write_json(full / "diagnostics.json", changed)
    with pytest.raises(ValueError, match="changed after binding"):
        verify_pivae_model(tmp_path, manifest, "full", cfg)


def test_hybrid_tree_cannot_accept_another_bound_pivae_stage(tmp_path):
    (tmp_path / "pivae").mkdir()
    source = tmp_path / "pivae" / "manifest.json"
    source.write_text("first independently bound piVAE stage")
    tree = {"models": {"hybrid_t": {}}, "pivae_manifest_sha256": file_hash(source)}
    verify_xgboost_pivae_binding(tmp_path, tree)
    source.write_text("another independently bound piVAE stage")
    with pytest.raises(ValueError, match="Hybrid tree"):
        verify_xgboost_pivae_binding(tmp_path, tree)
    with pytest.raises(ValueError, match="unbound"):
        verify_xgboost_pivae_binding(tmp_path, {"models": {"hybrid_t": {}}})
    verify_xgboost_pivae_binding(tmp_path, {"models": {"xgb_q": {}}})


def test_failed_fold_prevents_tree_fitting_and_output_creation(cfg, tmp_path, monkeypatch):
    import pivae_hybrid.stage_two as stage
    monkeypatch.setattr(stage, "verify_training_source", lambda _: None)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "train_simulation.csv"
    synthetic_data().to_csv(raw, index=False)
    cfg["data_dir"] = str(data_dir)
    run = tmp_path / "run"
    root = run / "pivae"
    models = {}
    for name in ["full", *(f"fold_{fold}" for fold in range(cfg["crossfit"]["folds"]))]:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "neural.pt").write_bytes(b"neural")
        (directory / "posterior.npz").write_bytes(b"posterior")
        diagnostics = good_diagnostics()
        diagnostics["quality_passed"] = name != "fold_0"
        write_json(directory / "diagnostics.json", diagnostics)
        models[name] = model_hashes(directory)
    write_json(root / "manifest.json", {"training_signature": training_signature(cfg),
               "features": cfg["features"], "training_sha256": file_hash(raw), "models": models})
    with pytest.raises(ValueError, match="quality gate failed"):
        stage.train_xgboost(cfg, run)
    assert not (run / "xgboost").exists()


@pytest.mark.parametrize("stage_name", ["pivae", "xgboost"])
def test_retry_does_not_overwrite_incomplete_stage_logs(cfg, tmp_path, stage_name):
    from pivae_hybrid.stage_one import train_pivae
    from pivae_hybrid.stage_two import train_xgboost
    root = tmp_path / stage_name
    root.mkdir()
    log = root / "unfinished-training.log"
    log.write_text("preserve evidence from interrupted training")
    function = train_pivae if stage_name == "pivae" else train_xgboost
    with pytest.raises(ValueError, match="preserve logs"):
        function(cfg, tmp_path)
    assert log.read_text() == "preserve evidence from interrupted training"
