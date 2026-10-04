# Data

The paper uses five-second measurements from the Institute of New Energy Systems (InES), Germany. The measurement files are not included. Obtain authorized copies through the project or data provider; the available materials contain no public download link or redistribution grant.

Pass the directory containing both CSVs with `--data-dir`.

| File | Rows | Dates |
|---|---:|---|
| `train_simulation.csv` | 130,048 | 21–27 and 29 August 2024; first and last days partial |
| `test_simulation.csv` | 17,280 | 28 August 2024, 00:00:00–23:59:55 |

The supplied total is 147,328 rows, 68 fewer than the paper reports. Timestamps are ordered, with no duplicates or overlap between files. Numeric values are finite. The reconstruction preserves the supplied split and rows.

## Inputs

The ten predictors are:

```text
T_in_1C, T_amb_outdoorC, T_vent_1C, T_IR_Tracker,
G_trackerWm, G_diffuseWm, IR_TrackerWm, v_wind_speedms,
zenith, azimuth
```

Training and evaluation also use `Datum`, `Q_1kW` and `T_out_1C`. New-input prediction requires chronological, unique `Datum` timestamps and the ten numeric predictors; targets may be omitted.

Measured Q, outlet temperature and `dT_1K` are excluded from predictors because `Q = m_flow × cp × (T_out − T_in) / 3600` in these files. The complete 22-column schema is in [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md).

## Original file hashes

```text
train_simulation.csv  2c63259c638777b0c988ff4919847bc345ceb8489cefb7ac95e54dd73ee2433f
test_simulation.csv   437dfdeb2c9e02476cdfcaf2a8fafc768be5ab4a3abab099fc3903cde7b7f319
```

These hashes identify the original split used for the saved reconstruction. New datasets need their own configuration and experiment record.

The only CSV in this repository contains 12 aggregate model/split scores. Raw measurements, row-level predictions, model weights, posterior samples, notebook outputs and measurement-derived images stay local. Any further distribution depends on the data-use agreement.
