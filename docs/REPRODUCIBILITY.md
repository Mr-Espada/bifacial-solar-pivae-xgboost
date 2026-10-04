# Reproducibility and scientific review

Reviewed 4 October 2026. **The published numerical experiment is not reproduced.** The retained October 2026 reconstruction's saved predictions, metrics and posterior exports have been checked independently. No full neural/HMC training was rerun during this audit or cleanup, and no paper result was replaced with a new score.

## Three separate records

| Record | What can be established | What remains missing |
|---|---|---|
| Original published work | Paper, five historical artifacts, source structure, raw split and notebook outputs | Final-Q training version, original XGBoost/simulator source, historical checkpoints/predictions/loss logs/settings and figure lineage |
| 2026 reconstruction and cleanup | Explicit configs; compatible saved models/draws; all-row frozen inference and score regeneration; source/data hashes; added maintenance safeguards | Independent full-training rerun; original author settings; broad generalization or full uncertainty calibration |
| Future improved experiments | A documented research agenda | No new results are claimed for those experiments |

The original archive script targets mass flow while the paper targets Q. It fits Stan on test labels and uses test loss for selection. Its outputs cannot establish a clean held-out benchmark. The original notebook applies statistics from concatenated train and test and needs missing `mcmc_1000.csv`. Historical originals are retained unchanged privately; public historical code is evidence, not an executable reproduction recipe.

## Data access and checks

See [DATA.md](DATA.md). Obtain authorized copies of the two original CSVs; they are not in this repository. Supply `--data-dir` explicitly. File hashes, 22-column schema, counts, dates, duplicate/overlap checks and physical identities are in [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) and `SOURCE_INVENTORY.json`.

Training is eight days (21–27 and 29 August 2024), 130,048 rows. Test is all 17,280 rows of 28 August. The total is 147,328, not the paper's 147,396. Training after the test date makes this held-out-day reconstruction, not causal forecasting. No cleaning/acquisition or upstream generation driver is available. Present CSV loading rejects nonfinite values, duplicate/unordered timestamps and non-allowlisted predictors; it does not silently impute or filter.

## Installation and execution

Use Python 3.10–3.12. The retained run and validation used Python 3.10.16 and these declared numerical pins: NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0, PyTorch 2.5.1, xgboost-cpu 3.0.2, CmdStanPy 1.2.5 and pytest 8.3.5. The retained environment has an unrecorded transitive mismatch: SymPy 1.13.3 versus PyTorch's required 1.13.1. The candidate explicitly pins SymPy 1.13.1; final compatibility validation uses an isolated overlay with that version and leaves the old environment untouched. This matters for environment reproducibility even though the checked saved predictions match. CmdStan 2.37.0 needs a C++ compiler and Make for new fits. The pinned CPU tree distribution may need a platform-specific substitution; alternative platforms are unverified.

From the repository root, follow the README installation block, then:

```bash
python -m pytest -q
python -m pivae_hybrid.cli audit --config configs/reconstruction.json --data-dir /path/to/authorized-data
python -m pivae_hybrid.cli run-all --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/new-reconstruction
```

This starts a **new** experiment with five complete πVAE fits. It is not the original author experiment. The short profile in `configs/verification.json` uses all raw rows but 80 epochs and smaller networks/two chains/300 trees; it also remains a separate experiment. Neither is an instant smoke test. Do not reuse a completed or interrupted stage with existing artifacts: use a fresh run directory so its logs remain intact.

To regenerate metrics/plots from a previously bound compatible run without fitting:

```bash
python -m pivae_hybrid.cli predict --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/checked-copy
python -m pivae_hybrid.cli evaluate --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/checked-copy
python -m pivae_hybrid.cli verify-run --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/checked-copy
```

For the retained 0.1.0 saved run, first use the explicit copy/import utility:

```bash
python scripts/import_saved_run.py --source-run /path/to/retained-run --source-package /path/to/original-reconstruction-package --data-dir /path/to/authorized-data --output-run runs/checked-copy
```

The original package must match the run's saved source hashes. Neural architecture/training/config and Stan source must match the current compatible implementation byte for byte. The utility first verifies the retained posterior quality reports, then copies the run, binds neural/posterior/diagnostic sets and metric inputs, binds the hybrid tree to its πVAE-stage manifest, verifies all serialized test outputs, and retains the original code manifest. Binding is **post hoc**, not recovered historical authentication. It neither retrains nor restores the lost eleven optimizer logs per fit. Its copy can be evaluated; changed-source fitting continuation is refused. Use a new import destination for the finalized bindings; do not silently revise the earlier audited copy.

## Leakage, metrics and scientific fairness

Each first-stage fold independently fits its own input/Q scaling, centers, network and training-label posterior. All rows of the predicted day are excluded. Stage two sees only predicted out-of-fold Q plus allowlisted inputs. The full-training first stage predicts the test day with no target argument. Trees have fixed parameters and no test `eval_set`/early stopping. These controls are 2026 reconstruction decisions, not retrospective proof of paper methodology.

Metrics score all rows in physical units. R² is 1−SSE/SST, Pearson is linear correlation, Spearman is rank correlation, MAE is mean absolute error, RMSE is root mean squared error. Constant/undefined scores are null; invalid values are rejected. Correlation is not predictive accuracy. Compare Q models in kW and outlet models in °C; comparing their R² as if they were the same task is not a fair ablation. Plot thinning does not thin scoring.

First-stage `train_oof` Q scores are held-out-day predictions. `train_fit` is in-sample. `train_stage2_fit_oof_q` has out-of-fold inputs but in-sample outlet fitting; it is not nested out-of-fold outlet evaluation. Direct/stacked trees use comparable tree settings, but that does not equalize total pipeline capacity, tuning effort or computational cost. The available one-day test cannot resolve the small hybrid gain reliably.

## Independently checked numerical evidence

All 12 saved metric records recalculated within 1e-12; all five test model rows match. The saved full-day predictions, including interval endpoints, exactly match fresh frozen inference after the original CSV serialization. All-row input-target deletion/poisoning leaves predictions bit-identical. All five z/beta/sigma archives exactly match raw Stan CSVs; decoder transfer error is below 5e-16. The original 19 tests passed; the maintained candidate has additional artifact/log regression tests. Current validation counts and command results are recorded in [VALIDATION.md](VALIDATION.md).

Retained 2026 test outlet RMSE: direct XGBoost .171045 °C; πVAE-Q stack .166222 °C; XGBoost-Q stack .192972 °C. πVAE Q RMSE .072010 kW, versus direct tree Q .052557 kW. The .004823 °C hybrid gain is one-day evidence, not general superiority. An exploratory training-slope energy-balance diagnostic using existing tree-Q predictions gives .161861 °C; it was review-motivated, not a pre-specified benchmark. Full results remain labeled reconstruction results in `results/reconstruction-2026-10-03/`.

The 95% observation predictive interval covers 94.473% of the test day but only 42.64% around noon and 65% in the following hour. Latent credible interval coverage of observed Q is 5.150%; that is not a nominal noisy-observation coverage test. Residual lag-one correlations .958/.981 show strong dependence. An iid Gaussian, single-sigma likelihood omits temporal/heteroscedastic noise; fixed neural weights omit their uncertainty; no Q uncertainty is propagated into outlet intervals.

## Posterior interpretation and implementation

The augmented-QR likelihood preserves all fitting observations and the correct sigma normalization. Existing numerical tests check equivalence to full SSE, including rank-deficient bases. Stan samples latent z/sigma through a fixed trained decoder, not all network parameters. Thirty-two replicas learn one observed Q function; they are not independent prior-function realizations as in foundational πVAE. Using Q to learn the map/decoder and condition the latent posterior is a data-dependent conditional approximation.

The long retained run has max Rhat 1.00523, min bulk ESS 2180.04, min tail ESS 1661.2, and no reported divergences/depth hits. Every relevant summary is finite. All chains start near one high-density MAP initialization: good diagnostics do not establish global mode probability or exploration. Keep optimizer starts distinct from distinct modes. The historical profile's 1.05 Rhat gate is documented, not silently tightened to change profile signatures; [Stan guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html) motivates stronger future gates.

Maintenance adds saved-model/diagnostic and prediction bindings, the hybrid tree's πVAE-stage binding, source-continuation checks, training-only raw-file hash checks, unique per-start log paths and interrupted-stage protection. Failed, missing or nonfinite posterior diagnostics now stop downstream fitting/inference; the gate rechecks the existing configured thresholds rather than trusting a stored flag alone. This changes future failure handling. It changes no scientific architecture, loss/likelihood, feature list, target, seed or configured hyperparameter. Original artifacts/scores are retained. No saved source code or old fitted artifact was rewritten. Passing the computational gate still does not establish predictive calibration.

## Reproduction boundary and next research

A new graduate student cannot regenerate the published Table 2 or exact Figs. 4–7 from this release alone. Author settings and original final artifacts are missing; data redistribution is unapproved; private fitted reconstruction bundles are excluded. With separately authorized data and compatible saved reconstruction artifacts, evaluation and prediction can be repeated without retraining. With data and a working toolchain, new reconstruction training can run; exact cross-platform or historical author metrics are not promised.

Future research should evaluate complete outlet pipelines in grouped outer folds with inner Q cross-fitting, multiple seeds, physical/deterministic/basis-only ablations, runtime/measurement-resolution context, fresh external days and block-aware uncertainty in comparisons. Globally precomputed Q is unsafe for outer validation if it can incorporate outer-held-out labels. Study temporal/heteroscedastic noise, neural uncertainty, mode sensitivity and the prior-function assumption. Further model selection on already-inspected August 28 requires a new independent test set for unbiased new claims.
