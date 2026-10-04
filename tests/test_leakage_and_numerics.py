from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from pivae_hybrid.config import PAPER_FEATURES, PREDICTED_Q, validate_config, load_config
from pivae_hybrid.data import read_inputs, add_predicted_q, identity_frame, day_folds
from pivae_hybrid.evaluation import regression_metrics, interval_metrics
from pivae_hybrid.model import PIVAE, calculate_loss
from pivae_hybrid.posterior import compress_gaussian_likelihood, predict_posterior
from pivae_hybrid.training import fit_neural, load_neural


@pytest.fixture
def cfg():
    path = Path(__file__).parents[1] / "configs" / "verification.json"
    config = load_config(path)
    config["pivae"].update(epochs=1, n_centers=8, beta_dim=8, latent_dim=3,
                           feature_hidden=[8, 8], vae_hidden=[8, 8], coefficient_replicas=4,
                           batch_size=16, threads=1)
    config["crossfit"]["folds"] = 2
    config["xgboost"].update(n_estimators=3, max_depth=2, n_jobs=1)
    return config


def synthetic_data():
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(80, len(PAPER_FEATURES))), columns=PAPER_FEATURES)
    frame["Datum"] = [str(pd.Timestamp(f"2024-08-{day:02}") + pd.Timedelta(seconds=i*5))
                      for day in (21, 22, 23, 24) for i in range(20)]
    frame["Q_1kW"] = 0.5*frame.T_in_1C + .1*frame.G_trackerWm
    frame["T_out_1C"] = frame.T_in_1C + 3*frame.Q_1kW
    frame["dT_1K"] = frame.T_out_1C - frame.T_in_1C
    return frame


@pytest.mark.parametrize("forbidden", ["T_out_1C", "dT_1K", "Q_1kW", PREDICTED_Q, "cp_1kJkgK", "rho_1kgl", "SecondsSinceStart", "T_mass_flow_1C"])
def test_config_rejects_target_or_unspecified_features(cfg, forbidden):
    cfg["features"].append(forbidden)
    with pytest.raises(ValueError, match="Unsafe"):
        validate_config(cfg)


def test_inference_reader_never_requests_target_columns(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    synthetic_data().to_csv(source, index=False)
    real_read = pd.read_csv
    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs["usecols"])
        return real_read(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", spy)
    before = read_inputs(source, list(PAPER_FEATURES))
    data = synthetic_data()
    data[["Q_1kW", "T_out_1C", "dT_1K"]] = np.nan
    data.to_csv(source, index=False)
    after = read_inputs(source, list(PAPER_FEATURES))
    pd.testing.assert_frame_equal(before, after)
    assert all(not set(cols) & {"Q_1kW", "T_out_1C", "dT_1K"} for cols in calls)


def test_whole_day_folds_are_disjoint_and_exhaustive():
    data = synthetic_data()
    data.insert(0, "row_id", np.arange(len(data)))
    counts = np.zeros(len(data), int)
    for _, fit, held, manifest in day_folds(data, 2):
        assert not set(manifest["fit_days"]) & set(manifest["prediction_days"])
        assert not set(fit) & set(held)
        counts[held] += 1
    np.testing.assert_array_equal(counts, np.ones(len(data)))


def test_stack_accepts_only_aligned_predicted_q():
    data = synthetic_data()
    data.insert(0, "row_id", np.arange(len(data)))
    predictions = identity_frame(data)
    predictions[PREDICTED_Q] = 0.0
    inputs = add_predicted_q(data, list(PAPER_FEATURES), predictions)
    assert list(inputs) == [*PAPER_FEATURES, PREDICTED_Q]
    assert not set(inputs) & {"Q_1kW", "T_out_1C", "dT_1K"}
    with pytest.raises(ValueError, match="align"):
        add_predicted_q(data, list(PAPER_FEATURES), predictions.iloc[::-1])


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_qr_reduction_equals_full_gaussian_likelihood(rank_deficient):
    rng = np.random.default_rng(2026)
    phi = rng.normal(size=(257, 12))
    if rank_deficient:
        phi[:, 5] = phi[:, 2]
    truth = rng.normal(size=257)
    reduced = compress_gaussian_likelihood(phi, truth)
    for _ in range(8):
        beta = rng.normal(size=12)
        full_sse = np.sum((phi @ beta - truth)**2)
        qr_sse = np.sum((reduced["R"] @ beta - reduced["qty"])**2) + reduced["residual_sse"]
        assert qr_sse == pytest.approx(full_sse, rel=1e-12, abs=1e-10)


def test_metrics_and_interval_units_are_correct():
    metrics = regression_metrics([1, 2, 4], [1, 3, 3])
    assert metrics["r2"] == pytest.approx(4/7)
    assert metrics["mae"] == pytest.approx(2/3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2/3))
    assert metrics["pearson"] == pytest.approx(np.corrcoef([1, 2, 4], [1, 3, 3])[0, 1])
    interval = interval_metrics([1, 2, 4], [0, 3, 3], [2, 4, 5])
    assert interval["coverage"] == pytest.approx(2/3)
    assert interval["mean_width"] == pytest.approx(5/3)
    assert regression_metrics([1, 1], [1, 1])["pearson"] is None


def test_latent_credible_and_predictive_intervals_are_distinct():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x
    class Model:
        phi = Identity()
    checkpoint = {"x_mean": [0, 0], "x_scale": [1, 1], "y_mean": 10.0, "y_scale": 2.0}
    posterior = {"beta": np.ones((4000, 2)), "sigma": np.ones(4000)}
    prediction = predict_posterior(Model(), checkpoint, [[1, 2], [2, 3]], posterior, seed=12)
    np.testing.assert_allclose(prediction[PREDICTED_Q], [16, 20])
    np.testing.assert_allclose(prediction["q_ci_low"], prediction["q_ci_high"])
    assert np.all(prediction["q_pi_high"] - prediction["q_pi_low"] > 7)


def test_coefficients_are_persistent_for_partial_minibatches(cfg):
    torch.manual_seed(12)
    model = PIVAE(len(PAPER_FEATURES), cfg["pivae"]).eval()
    x = torch.randn(7, len(PAPER_FEATURES))
    whole = model(x)[1]
    partial = model(x[:3])[1]
    assert whole.shape == (4, 7)
    assert partial.shape == (4, 3)
    torch.testing.assert_close(partial, whole[:, :3], atol=1e-6, rtol=1e-6)
    direct, decoded, mu, logvar = model(x)
    assert torch.isfinite(calculate_loss(torch.randn(7), direct, decoded, mu, logvar, .001)[0])


def test_training_only_scaling_and_checkpoint_roundtrip(cfg, tmp_path):
    data = synthetic_data()
    x = data[list(PAPER_FEATURES)].to_numpy()
    model, checkpoint = fit_neural(x[:40], data.Q_1kW.to_numpy()[:40], list(PAPER_FEATURES), cfg, 9, tmp_path)
    np.testing.assert_allclose(checkpoint["x_mean"], x[:40].mean(axis=0))
    assert checkpoint["y_mean"] == pytest.approx(data.Q_1kW.iloc[:40].mean())
    loaded, restored = load_neural(tmp_path / "neural.pt")
    assert restored["features"] == list(PAPER_FEATURES)
    xs = torch.randn(5, len(PAPER_FEATURES))
    torch.testing.assert_close(model(xs)[1], loaded(xs)[1])
