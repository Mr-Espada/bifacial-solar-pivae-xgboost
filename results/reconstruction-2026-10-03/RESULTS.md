# Reconstructed experiment results

Profile: `reconstruction_500_epochs`. Seed: `20261003`.

These are October 2026 reconstruction results. The paper's final settings and fitted models are unavailable, so its reported scores could not be reproduced.

Training uses the original 130,048 rows on eight dates. Scoring uses all 17,280 observations on the held-out August 28, 2024. Training includes August 29 as in the paper; this is held-out-day simulation, not future forecasting.

## Held-out metrics

| Model | Target unit | R2 | Pearson | Spearman | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| piVAE Q | kW | 0.994684 | 0.997454 | 0.992255 | 0.042240 | 0.072010 |
| XGBoost Q | kW | 0.997169 | 0.998645 | 0.993856 | 0.031508 | 0.052557 |
| XGBoost outlet (direct) | degC | 0.997849 | 0.998972 | 0.995728 | 0.106478 | 0.171045 |
| piVAE Q -> XGBoost outlet | degC | 0.997968 | 0.998988 | 0.996359 | 0.103431 | 0.166222 |
| XGBoost Q -> XGBoost outlet | degC | 0.997262 | 0.998639 | 0.995642 | 0.114499 | 0.192972 |

Hybrid minus direct outlet MAE: -0.003047 degC; RMSE: -0.004823 degC. Negative values favor the hybrid.

Q-model and outlet-model errors have different units; compare piVAE with XGBoost Q, and compare the three outlet models with each other.

## Posterior intervals on held-out measured Q

| 95% interval | Empirical coverage | Mean width (kW) | Median width (kW) |
|---|---:|---:|---:|
| Latent function credible | 5.150% | 0.002407 | 0.002244 |
| Observation posterior predictive | 94.473% | 0.305646 | 0.305655 |

The latent credible interval conditions on the learned feature map/decoder. Comparing it with noisy measured Q does not establish nominal observation coverage. Posterior predictive intervals also include the Stan observation noise. Neither interval includes neural-weight uncertainty or accounts fully for temporal dependence.

## Settings and diagnostics

Neural fits: 500 fixed epochs, 128 RBF centers, feature hidden widths [128, 128], beta dimension 64, latent dimension 24, VAE widths [128, 64], 32 persistent coefficient replicas, batch size 2048, learning rate 0.0003, KL weight 0.001.

Stack training: 4 whole-day folds; each scaler, center set, feature map, VAE and HMC likelihood excludes the predicted training day's rows. Final test Q uses a separate fit on all training rows.

Each HMC fit: 4 chains, 1000 warmup and 1000 retained draws per chain. Gaussian likelihood uses all fitting observations via exact QR statistics; no observation subsampling.

HMC initialization: multistart_map; metric: dense_e. Multi-start initialization uses only the fitting likelihood and is an optimization aid, not the point predictor. All prediction/interval samples still come from HMC. `initialization.json` records discovered local modes and the selected density. Convergence gates describe exploration of the initialized mode and do not certify global exploration of every mode.

| Posterior | Max Rhat | Min bulk ESS | Divergences | Depth hits | Quality gate |
|---|---:|---:|---:|---:|---|
| fold_0 | 1.0052 | 4776.7 | 0 | 0 | pass |
| fold_1 | 1.0051 | 4227.7 | 0 | 0 | pass |
| fold_2 | 1.0049 | 4057.8 | 0 | 0 | pass |
| fold_3 | 1.0037 | 2180.0 | 0 | 0 | pass |
| full | 1.0030 | 2765.8 | 0 | 0 | pass |

Quality gate requires Rhat at or below the configured threshold, minimum bulk ESS, no divergences/depth saturation and BFMI above 0.3 in each chain. Exact diagnostics and CmdStan reports are saved per fit.

## Reproduction artifacts

`resolved_config.json`, `environment.json`, stage manifests, original CSV hashes, fitted neural/tree models, posterior samples and Stan chain CSVs identify this run. `metrics.json` and `metrics.csv` include separate training-fit and first-stage out-of-fold scores. Outlet training scores fit stage two using out-of-fold Q; they are not out-of-fold outlet scores.

Run `python -m pivae_hybrid.cli evaluate --config configs/reconstruction.json --data-dir /path/to/authorized-data --run-dir runs/checked-copy` to regenerate scores and plots from a compatible saved run without fitting. See the [README](../../README.md) for stage commands and the [research audit](../../docs/RESEARCH_AUDIT.md) for paper/source discrepancies.

Plots show all held-out rows; scatter and combined-record lines use deterministic display thinning only. No metrics are thinned. XGBoost importance is training gain, not a causal explanation.
