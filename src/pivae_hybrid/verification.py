"""Integration proof that absent/poisoned target columns cannot affect inference."""

import csv
import tempfile
from pathlib import Path
import numpy as np
from .inference import predict_csv
from .data import write_json, verify_raw


def verify_inference_invariance(cfg, run_dir, baseline_only=False):
    source = Path(cfg["data_dir"]) / "test_simulation.csv"
    expected = predict_csv(cfg, run_dir, source, include_pivae=not baseline_only)
    # Preserve original input strings so changes test labels, not float serialization.
    with tempfile.TemporaryDirectory(prefix="target_invariance_", dir=run_dir) as scratch:
        absent = Path(scratch) / "without_targets.csv"
        poisoned = Path(scratch) / "poisoned_targets.csv"
        with source.open(newline="") as stream, absent.open("w", newline="") as a, poisoned.open("w", newline="") as b:
            reader = csv.DictReader(stream)
            keep = ["Datum", *cfg["features"]]
            aw = csv.DictWriter(a, fieldnames=keep)
            bw = csv.DictWriter(b, fieldnames=reader.fieldnames)
            aw.writeheader()
            bw.writeheader()
            for row in reader:
                aw.writerow({key: row[key] for key in keep})
                for key in ("Q_1kW", "T_out_1C", "dT_1K"):
                    row[key] = "999999999"
                bw.writerow(row)
        checks = {}
        for name, path in (("targets_absent", absent), ("targets_poisoned", poisoned)):
            actual = predict_csv(cfg, run_dir, path, include_pivae=not baseline_only)
            if list(actual.columns) != list(expected.columns):
                raise AssertionError("Inference changed its output schema")
            equal = all(np.array_equal(actual[col].to_numpy(), expected[col].to_numpy()) for col in expected)
            if not equal:
                raise AssertionError(f"Inference changed when {name}")
            checks[name] = {"bit_identical": True, "rows": len(actual), "columns": list(actual.columns)}
    checks["raw_csv_sha256"] = verify_raw(cfg["data_dir"])
    write_json(Path(run_dir) / "inference_invariance.json", checks)
    return checks
