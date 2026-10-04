# Research audit

Reviewed on 4 October 2026 against the paper, historical source, original CSV split and saved reconstruction. The paper's final numerical experiment is not reproducible from the available materials. Saved October 2026 reconstruction results were checked without retraining.

## Historical source

The original archive contains a PyTorch training script, Stan model, analysis notebook and two CSVs. It has no XGBoost implementation, CARNOT/Simulink model, final checkpoint, retained posterior CSV, experiment log or dependency file. The notebook references a missing `mcmc_1000.csv`.

The neural code adapts [MLGlobalHealth/pi-vae](https://github.com/MLGlobalHealth/pi-vae), reviewed at commit `14de0377500ce175115a6509a802e473c3466a54`. Encoder, Decoder, VAE and the coefficient loss match after AST formatting normalization. The upstream MIT notice is retained in [third_party/pi-vae/LICENSE](../third_party/pi-vae/LICENSE).

The historical script uses nine inputs, omits inlet temperature and targets `m_flow_1kgh`. It plans 10,000 epochs, batch size 32 and Adam learning rate .0003. RBF center count is half the test row count, or 8,640; inputs are unscaled. Its 32 coefficient rows are attached to shuffled minibatch slots rather than persistent functions.

Every 200 epochs, including epoch zero, Stan conditions on all test labels. Test loss also selects the checkpoint. HMC uses four chains, 200 warmup and 1,000 retained draws, without recorded seeds. Importing the script starts logging, Stan compilation and training. This source cannot establish a clean held-out benchmark.

The historical encoder variable `sd` represents log variance; Stan's `sigma2` is a standard deviation. The plotted Stan output includes observation noise, despite being called a credible interval. The notebook combines train and test for normalization. Its 21 code cells and saved figures show a Q-oriented analysis, but no metric table or tree fitting. Non-monotonic execution counters and the missing posterior prevent recovery of the complete workflow.

## Data and physical relationships

The paper attributes the measurements to InES, Germany. Acquisition, sensor calibration, cleaning, timezone convention and generation of derived columns are not documented in the retained source. Filenames containing “simulation” do not establish synthetic provenance.

The 22 columns are:

```text
T_in_1C, T_out_1C, T_mass_flow_1C, T_amb_outdoorC, T_vent_1C,
Vol_flow_1lh, v_wind_speedms, G_trackerWm, G_diffuseWm, IR_TrackerWm,
T_IR_Tracker, cp_1kJkgK, m_flow_1kgh, rho_1kgl, dT_1K, Q_1kW,
SecondsSinceStart, Datum, zenith, azimuth, direct_rad_calc, diffuse_rad_calc
```

Training covers 21 August 2024 04:00:05 to 29 August 16:37:20, excluding 28 August: 14,399 rows on the 21st, 17,280 on each of the 22nd–27th, and 11,969 on the 29th. Test contains all 17,280 five-second rows of 28 August. There are no duplicate, unordered or shared timestamps, or missing/nonfinite numeric values. The total is 147,328, 68 fewer than the paper's 147,396. Original hashes are in [DATA.md](DATA.md).

`dT = T_out − T_in` holds within 1.00000001e-6 K, and `Q = m_flow × cp × dT / 3600` within 8.358354e-7 kW. Measured Q, outlet and dT can therefore encode the target algebraically. The reconstruction excludes them from predictors. Other omitted columns have uncertain availability at inference time; their exclusion does not establish that each is intrinsically leaky.

## Reconstruction methods

The reconstruction follows the paper's Fig. 3: ten measured inputs → πVAE-predicted Q → XGBoost outlet temperature. The tree receives scalar Q, rather than latent z or coefficients β. Scaling, center initialization, neural fitting and Stan conditioning use each training fold only. Four whole-day folds generate training Q; a separate all-training fit supplies test Q. Because training includes 29 August, the test represents a held-out day rather than future forecasting.

The full profile uses 128 centers, feature-map widths 128/128, 64 coefficients, 24 latent dimensions, VAE widths 128/64 and 32 persistent replicas of one observed Q function. Training uses 500 fixed epochs, batch 2,048, Adam .0003, RBF factor .25, KL weight .001 and gradient clipping 10. Seeds are fixed and deterministic PyTorch operations enabled. These choices were made for reconstruction; they are not recovered final paper settings.

Stan keeps the learned decoder fixed and samples latent z and observation sigma. An augmented-QR Gaussian likelihood uses all fitting rows and retains the −n log(sigma) normalization. The sigma prior is Normal(.025, .01), bounded below by .001, in standardized-Q units. Its physical scale differs from the historical raw-target prior. Twelve training-only optimizer starts select an initialization; four chains then use 1,000 warmup/1,000 retained draws, a dense metric, adapt_delta .95 and maximum tree depth 12.

XGBoost uses 500 histogram trees, depth 6, learning rate .05, row/column sampling .9, lambda 1 and seed 20261003. Direct Q, direct outlet, πVAE-Q outlet and XGBoost-Q outlet models are compared without test early stopping. The smaller verification profile is a separate reconstruction configuration.

## Paper-to-source mapping

| Paper item | Available source or output | Finding |
|---|---|---|
| 147,396 observations, p.172 | Two original CSVs | 147,328 rows; acquisition and cleaning missing. |
| Eq. 1, integrated energy | CSV target `Q_1kW` | The implementation models instantaneous power; the paper mixes power and energy units. |
| Table 1 | Original CSV schema | Named variables are present; physical calibration is unknown. |
| Figs. 1–2, Simulink/CARNOT | No simulator source or outputs | Classical comparison cannot be regenerated. |
| Fig. 3(a), πVAE-Q with ten inputs | Historical RBF/VAE/Stan topology | Historical script targets mass flow and omits inlet; reconstruction implements Q with ten inputs. |
| Eqs. 2–4, decoder and posterior mean | Python/Stan source | Reconstruction computes E[d(z)]·Φ(x); the historical posterior is missing. |
| Fig. 3(b), Eq. 5, tree outlet stage | No historical tree code or model | Tree stage was added in the reconstruction. |
| Held-out 28 August, p.175 | Original CSV split | Split matches, but historical Stan conditions on the test labels. |
| Fig. 4, CARNOT comparison | No simulator outputs | Axes show 2023 while the methods and data use 2024. |
| Fig. 5(a), loss over about 500 epochs | Historical logging code, no loss log | Historical script plans 10,000 epochs; reconstruction loss plots are new. |
| Fig. 5(b), tree importance | No historical tree or gain file | Reconstruction importance plots cannot establish original figure provenance. |
| Table 2 | Published scores only | Targets/splits are unclear under “Training Results”; p.177 also describes a performance table as awaiting. |
| Fig. 6, Q predictions and bands | Historical notebook plots; missing posterior | Exact figure match and interval interpretation are unresolved. |
| Fig. 7(a), full-period Q | Notebook measurement plots, no complete predictions | Exact historical prediction figure is unlinked. |
| Fig. 7(b), full-period outlet | Measured outlet plot, no historical tree predictions | Exact historical prediction figure is unlinked. |

The abstract refers to latent features, while the methods and diagram feed predicted Q to XGBoost. Reconstruction follows the latter. Visual resemblance to a figure does not establish its numerical source.

## Checked reconstruction results

All 12 saved model/split records matched independent sklearn/scipy recalculation within 1e-12. Frozen inference matched all 17,280 serialized test rows, including interval endpoints. Removing or poisoning target columns left predictions unchanged. All five posterior exports matched raw Stan draws; decoder transfer error was below 5e-16. Current package checks are in [VALIDATION.md](VALIDATION.md).

| Model and target | R² | MAE | RMSE |
|---|---:|---:|---:|
| πVAE, Q in kW | .994684 | .042240 | .072010 |
| Direct XGBoost, Q in kW | .997169 | .031508 | .052557 |
| Direct XGBoost, outlet °C | .997849 | .106478 | .171045 |
| πVAE-Q → XGBoost, outlet °C | .997968 | .103431 | .166222 |
| XGBoost-Q → XGBoost, outlet °C | .997262 | .114499 | .192972 |

The hybrid's outlet RMSE improvement over direct trees is .004823 °C, or 2.82%, on one day. Direct trees perform better for Q. A later diagnostic fitted `T_out = T_in + k × predicted_Q` on training labels, with k = 3.079510965. Using existing tree-Q predictions, it reached .161861 °C test RMSE. This was an exploratory post-publication comparison, not a historical benchmark.

The 95% observation interval covers 94.473% of the test day at average width .305646 kW, but only 42.64% around noon and 65% in the next hour. Latent interval coverage against observed Q is 5.150%, at width .002407 kW; that is not a nominal noisy-observation coverage test. Residual lag-one correlations are .95824 for πVAE Q and .98121 for hybrid outlet.

Across five fits, maximum Rhat is 1.00523, minimum bulk ESS 2180.04 and minimum tail ESS 1661.2, with no reported divergences or depth saturation. Chains start near one selected high-density initialization. Good within-mode diagnostics do not establish global mode exploration or predictive calibration. The configured 1.05 Rhat gate is unchanged; stronger gates should be tested in a new profile. See [Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html).

## Scientific limitations and next experiments

The five-second observations are strongly dependent and the test covers one day. A small RMSE difference does not establish broad superiority. Second-stage training scores fit outlet labels in sample even though Q inputs are out of fold. Tree capacity, tuning effort and pipeline cost are not equalized by using comparable tree settings.

The coefficient replicas describe one observed function, unlike the independent prior-function collection in foundational πVAE. Learning the map and decoder on Q before conditioning on Q gives a data-dependent conditional approximation. Fixed neural weights, one iid Gaussian noise scale, local HMC starts and absent downstream uncertainty limit interpretation.

Useful next comparisons include nested whole-day outlet validation, repeated seeds, physical and deterministic baselines, basis-only ablations, independent days, temporal noise and mode sensitivity. Globally precomputed Q can leak outer-fold labels into a nested comparison. Further model selection after inspecting 28 August needs a new independent test set. Unused GP imports and commented coefficient alternatives do not establish fitted historical comparators or a dated model-selection history.

Original source and scientific outputs remain archived locally. The public repository retains historical code and source-only notebook cells, aggregate reconstruction scores and the upstream notice. Measurement files, fitted artifacts and row-level outputs are excluded.
