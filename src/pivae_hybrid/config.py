"""Explicit configuration; no settings are inferred from held-out data."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

PAPER_FEATURES = (
    "T_in_1C", "T_amb_outdoorC", "T_vent_1C", "T_IR_Tracker",
    "G_trackerWm", "G_diffuseWm", "IR_TrackerWm", "v_wind_speedms",
    "zenith", "azimuth",
)
PREDICTED_Q = "predicted_Q_1kW"
TARGET_Q = "Q_1kW"
TARGET_T = "T_out_1C"


def validate_config(cfg):
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    features = cfg["features"]
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be a nonempty unique allowlist")
    unknown = set(features) - set(PAPER_FEATURES)
    if unknown:
        raise ValueError(f"Unsafe or unspecified predictors: {sorted(unknown)}")
    if cfg["crossfit"]["folds"] < 2:
        raise ValueError("Stacking requires at least two whole-day folds")
    for name in ("epochs", "batch_size", "n_centers", "beta_dim", "latent_dim", "coefficient_replicas", "threads"):
        if cfg["pivae"][name] < 1:
            raise ValueError(f"pivae.{name} must be positive")
    if cfg["posterior"]["chains"] < 2:
        raise ValueError("At least two HMC chains are required for diagnostics")
    if cfg["posterior"]["samples"] < 4 or cfg["posterior"]["warmup"] < 1:
        raise ValueError("Invalid posterior draw counts")
    if cfg["posterior"].get("metric", "diag_e") not in ("diag_e", "dense_e", "unit_e"):
        raise ValueError("Invalid HMC metric")
    if cfg["posterior"].get("initialization", "encoder") not in ("encoder", "multistart_map"):
        raise ValueError("Unknown posterior initialization")
    if cfg["posterior"].get("initialization") == "multistart_map":
        if cfg["posterior"]["map_starts"] < 2 or cfg["posterior"]["map_iters"] < 1:
            raise ValueError("Invalid multi-start initialization settings")
    if not 0 < cfg["interval_alpha"] < 1:
        raise ValueError("interval_alpha must lie in (0,1)")
    return cfg


def load_config(path, data_dir=None):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    root = path.parent.parent
    cfg["data_dir"] = str(Path(data_dir).resolve() if data_dir else (root / cfg["data_dir"]).resolve())
    return validate_config(cfg)


def training_signature(cfg):
    """Bind stage artifacts to the same fitting/prediction settings."""
    clean = deepcopy(cfg)
    clean.pop("profile", None)
    clean.pop("data_dir", None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()
