import json
from pathlib import Path
import numpy as np
import pandas as pd
from test_leakage_and_numerics import cfg, synthetic_data
from pivae_hybrid.config import PREDICTED_Q
from pivae_hybrid.data import read_inputs, day_folds
from pivae_hybrid.stage_one import train_pivae
from pivae_hybrid.stage_two import train_xgboost


def test_first_stage_neural_and_posterior_folds_never_read_test(cfg, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = synthetic_data()
    raw.to_csv(data_dir / "train_simulation.csv", index=False)
    cfg["data_dir"] = str(data_dir)
    # No test file exists: fitting must succeed without opening one.
    import pivae_hybrid.stage_one as stage
    # Synthetic fixture hashes are deliberately outside the public data contract.
    monkeypatch.setattr(stage, "verify_training_source", lambda _: None)
    monkeypatch.setattr(stage, "build_stan_model", lambda _: object())
    fitting_rows = []
    def conditional_stub(model, checkpoint, x_train, y_train, config, seed, output, stan_model):
        fitting_rows.append((x_train.copy(), y_train.copy()))
        result = {"beta": np.zeros((8, config["pivae"]["beta_dim"])), "sigma": np.ones(8)*.025,
                "z": np.zeros((8, config["pivae"]["latent_dim"]))}
        np.savez_compressed(output / "posterior.npz", **result)
        (output / "diagnostics.json").write_text(json.dumps({
            "quality_passed": True, "max_rhat": 1.0, "min_bulk_ess": 800.0,
            "bfmi_per_chain": [1.0, 1.0], "divergences": 0, "max_treedepth_hits": 0,
            "scope": "Synthetic stub; this test checks fitting-row boundaries only"}))
        return result
    monkeypatch.setattr(stage, "condition_posterior", conditional_stub)
    run = tmp_path / "run"
    train_pivae(cfg, run)
    train = read_inputs(data_dir / "train_simulation.csv", cfg["features"], target="Q_1kW")
    for (actual_x, actual_y), (_, fit, held, _) in zip(fitting_rows, day_folds(train, 2)):
        np.testing.assert_array_equal(actual_x, train[cfg["features"]].to_numpy()[fit])
        np.testing.assert_array_equal(actual_y, train.Q_1kW.to_numpy()[fit])
        assert not set(train.Datum.iloc[fit]) & set(train.Datum.iloc[held])
    np.testing.assert_array_equal(fitting_rows[-1][1], train.Q_1kW.to_numpy())
    oof = pd.read_csv(run / "pivae" / "q_train_oof.csv")
    assert len(oof) == len(raw) and oof[PREDICTED_Q].notna().all()
    # The stub tests the fitting boundary only; real Stan runs are verified separately.


def test_tree_training_does_not_need_or_read_a_test_file(cfg, tmp_path, monkeypatch):
    import pivae_hybrid.stage_two as stage
    monkeypatch.setattr(stage, "verify_training_source", lambda _: None)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    synthetic_data().to_csv(data_dir / "train_simulation.csv", index=False)
    cfg["data_dir"] = str(data_dir)
    run = tmp_path / "run"
    train_xgboost(cfg, run, baseline_only=True)
    manifest = json.loads((run / "xgboost" / "manifest.json").read_text())
    stack = manifest["models"]["xgb_q_t"]
    assert stack["features"] == [*cfg["features"], PREDICTED_Q]
    assert not set(stack["features"]) & {"Q_1kW", "T_out_1C", "dT_1K"}
    assert manifest["models"]["xgb_q"]["target"] == "Q_1kW"
    assert manifest["models"]["xgb_t"]["target"] == "T_out_1C"
