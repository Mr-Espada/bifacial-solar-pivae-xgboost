# Reproducibility

The saved October 2026 reconstruction can be checked with authorized data and compatible fitted artifacts. The paper's original numerical experiment cannot be recovered: final models, predictions, tree and simulator code, and complete settings are missing.

No full neural/HMC training was repeated during the repository review. The checked results came from frozen inference and recalculation of saved scores.

## Environment and data

Follow the [README setup](../README.md#setup). The checked environment used Linux, Python 3.10.16, the pinned Python dependencies and CmdStan 2.37.0. SymPy is pinned to 1.13.1 for compatibility with PyTorch 2.5.1. A fresh complete toolchain installation and other platforms have not been tested.

Provide the two original CSVs described in [DATA.md](DATA.md). They contain 130,048 training rows and the complete 17,280-row test day. File hashes and the split are checked by:

```bash
python -m pivae_hybrid.cli audit --config configs/reconstruction.json --data-dir /path/to/authorized-data
```

## New training

```bash
python -m pivae_hybrid.cli run-all --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
```

The full profile uses 500 epochs per fit, 128 RBF centers, feature-map widths 128/128, 64 coefficients, 24 latent dimensions, VAE widths 128/64 and 32 coefficient replicas. Batch size is 2,048, Adam learning rate .0003 and KL weight .001. Each of five fits uses twelve optimizer starts and four HMC chains, with 1,000 warmup and 1,000 retained draws per chain. Tree models use 500 depth-6 trees at learning rate .05 and seed 20261003.

These are reconstruction settings; the paper's final settings were not recovered. `configs/verification.json` uses smaller networks, 80 epochs, two chains and 300 trees. `run-all --baseline-only` runs the direct trees and tree-Q stack without neural/HMC fitting. Always use a fresh run directory.

To run stages separately, keep the same configuration, data directory and run directory throughout:

```bash
python -m pivae_hybrid.cli train-pivae --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
python -m pivae_hybrid.cli train-xgb --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
python -m pivae_hybrid.cli predict --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
python -m pivae_hybrid.cli evaluate --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
python -m pivae_hybrid.cli verify-run --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
```

`evaluate` regenerates scores and PNG/SVG figures without fitting. `verify-run` checks whether predictions change when input target columns are removed or poisoned. Predictions, posteriors, plots and models stay in the local run directory.

## Older saved runs

A retained 0.1.0 run needs an explicit import into a new directory:

```bash
python scripts/import_saved_run.py --source-run /path/to/retained-run --source-package /path/to/original-reconstruction-package --data-dir /path/to/authorized-data --output-run runs/checked-copy
```

The original package must match the run's source hashes, and its model, training, configuration and Stan files must match the compatible implementation. The import checks posterior quality, copies the run, adds artifact bindings and verifies serialized test predictions. It preserves the original source manifest. These added hashes do not recover historical authentication or missing optimizer logs.

After import, use the `predict`, `evaluate` and `verify-run` commands above with `runs/checked-copy`. Existing or interrupted fitting stages cannot be overwritten or continued with changed source.

## Evaluation

Scaling, center initialization, neural fitting and posterior conditioning use only each fold's fitting rows. Four whole-day folds produce training Q predictions, and a separate full-training fit supplies test Q. Trees receive out-of-fold predicted Q during training and never use test labels for early stopping.

Scores use all rows in physical units. Compare thermal-power models in kW and outlet models in °C. First-stage out-of-fold scores are held-out predictions; second-stage training scores still fit outlet labels in sample. Independent assessment of the complete outlet pipeline needs outer day-level folds.

The review checked all 12 saved metric records, full-day frozen predictions, target independence and all five posterior exports. Results and limitations are in [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md); execution details are in [VALIDATION.md](VALIDATION.md).

The historical notebook uses combined train/test statistics and is missing its posterior CSV. Historical source also conditions on test labels. Neither provides a clean reproduction of the published benchmark. New training cannot supply that missing historical record.
