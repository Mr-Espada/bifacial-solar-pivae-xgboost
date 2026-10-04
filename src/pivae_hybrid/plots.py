"""Deterministic scientific plots in physical units, independent of fitting."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def make_plots(cfg, root, test):
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(Path(cfg["data_dir"]) / "test_simulation.csv", usecols=["Datum", "Q_1kW", "T_out_1C"])
    times = pd.to_datetime(test.Datum)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=170)
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)

    def q_plot(mask, title, filename):
        fig, ax = plt.subplots(figsize=(12, 4))
        t = times[mask]
        ax.plot(t, truth.Q_1kW.to_numpy()[mask], color="black", lw=0.8, label="Measured Q")
        if "pivae_q" in test:
            ax.fill_between(t, test.q_pi_low.to_numpy()[mask], test.q_pi_high.to_numpy()[mask], color="#3977ae", alpha=.13, label="95% posterior predictive")
            ax.fill_between(t, test.q_ci_low.to_numpy()[mask], test.q_ci_high.to_numpy()[mask], color="#3977ae", alpha=.4, label="95% latent credible")
            ax.plot(t, test.pivae_q.to_numpy()[mask], color="#1565a4", lw=1, label="piVAE posterior mean")
        if "xgb_q" in test:
            ax.plot(t, test.xgb_q.to_numpy()[mask], color="#bd672c", lw=.8, alpha=.8, label="XGBoost Q")
        ax.set(title=title, ylabel="Thermal power (kW)", xlabel="August 28, 2024 time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=.18)
        finish(fig, filename)

    q_plot(np.ones(len(test), dtype=bool), "Held-out August 28: Q predictions and conditional uncertainty", "test_q_intervals")
    start, end = (pd.Timestamp(t) for t in cfg["plot_zoom"])
    mask = ((times >= start) & (times <= end)).to_numpy()
    if mask.any():
        q_plot(mask, "Held-out Q: fixed configured zoom window", "test_q_zoom")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, truth.T_out_1C, color="black", lw=1, label="Measured outlet")
    for col, name, color in (("xgb_t", "Direct XGBoost", "#858585"), ("hybrid_t", "piVAE Q -> XGBoost", "#1565a4"), ("xgb_q_t", "XGBoost Q -> XGBoost", "#bd672c")):
        if col in test:
            ax.plot(times, test[col], lw=.9, color=color, alpha=.85, label=name)
    ax.set(title="Held-out August 28: outlet temperature ablations", ylabel="Outlet temperature (degC)", xlabel="August 28, 2024 time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8)
    ax.grid(alpha=.18)
    finish(fig, "test_outlet")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    display = np.arange(0, len(test), max(1, len(test)//1500))
    for ax, target, columns, unit in ((axes[0], "Q_1kW", ("pivae_q", "xgb_q"), "kW"), (axes[1], "T_out_1C", ("xgb_t", "hybrid_t", "xgb_q_t"), "degC")):
        for col in columns:
            if col in test:
                ax.scatter(truth[target].iloc[display], test[col].iloc[display], s=3, alpha=.3, label=col)
        low, high = truth[target].min(), truth[target].max()
        ax.plot([low, high], [low, high], color="black", lw=.8)
        ax.set(xlabel=f"Measured ({unit})", ylabel=f"Predicted ({unit})", title=target)
        ax.legend(fontsize=8)
    finish(fig, "test_scatter")

    histories = sorted((root / "pivae").glob("*/loss.csv"))
    if histories:
        fig, ax = plt.subplots(figsize=(9, 4))
        for path in histories:
            history = pd.read_csv(path)
            ax.semilogy(history.epoch, history.loss, label=path.parent.name)
        ax.set(title="piVAE training-only loss (fixed epochs)", xlabel="Epoch", ylabel="Normalized reconstruction + KL loss")
        ax.legend()
        finish(fig, "training_loss")
    for path in sorted((root / "xgboost").glob("*_importance.csv")):
        importance = pd.read_csv(path).sort_values("gain")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance.feature, importance.gain, color="#3977ae")
        ax.set(xlabel="Training split gain", title=path.stem.replace("_", " "))
        finish(fig, path.stem)

    if (root / "pivae" / "q_train_fitted.csv").exists() and "hybrid_t" in test:
        train_q = pd.read_csv(root / "pivae" / "q_train_fitted.csv")
        train_t = pd.read_csv(root / "xgboost" / "train_predictions.csv")
        measured_train = pd.read_csv(Path(cfg["data_dir"]) / "train_simulation.csv", usecols=["Datum", "Q_1kW", "T_out_1C"])
        combined = measured_train.copy()
        combined["q_pred"] = train_q.predicted_Q_1kW
        combined["t_pred"] = train_t.hybrid_t
        test_combined = truth.copy()
        test_combined["q_pred"] = test.pivae_q
        test_combined["t_pred"] = test.hybrid_t
        combined = pd.concat([combined, test_combined]).sort_values("Datum").iloc[::20]
        dates = pd.to_datetime(combined.Datum)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax, target, pred, unit in ((axes[0], "Q_1kW", "q_pred", "kW"), (axes[1], "T_out_1C", "t_pred", "degC")):
            ax.plot(dates, combined[target], color="black", lw=.7, label="Measured")
            ax.plot(dates, combined[pred], color="#1565a4", lw=.8, alpha=.8, label="Reconstructed")
            ax.axvspan(pd.Timestamp("2024-08-28"), pd.Timestamp("2024-08-29"), color="#d7a845", alpha=.18, label="Held-out day")
            ax.set(ylabel=unit)
            ax.legend(ncol=3, fontsize=8)
        axes[0].set_title("Complete supplied record: training fits and held-out-day predictions")
        axes[1].set_xlabel("2024 date")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        finish(fig, "full_record")
