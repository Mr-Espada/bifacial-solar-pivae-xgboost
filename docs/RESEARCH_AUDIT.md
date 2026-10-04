# Research audit: published solar-collector study and retained reconstruction

Audit date: 4 October 2026. Recorded before creating the cleanup candidate. The 34 supplied files (165,289 bytes) were read and hashed; no original file was edited. This report distinguishes historical source, publication, October 2026 reconstruction, personal account, and inference. File ownership, archive timestamps, and paper authorship do not establish individual code authorship.

## Scope and evidence

The requested folder is `pi-vae-reconstruction-code`, containing three review/result documents and a 31-file Python package. It has no `.git` directory, Git history, remote, author signatures, dataset, historical notebook, trained model, or chain files. A complete path/size/SHA-256 inventory and verified pre-cleanup archive are retained privately. Complete trees are supplied in `docs/SOURCE_TREE.txt`.

Adjacent relevant evidence was inspected read-only: the original five-file `pi-vae.zip`, its retained code/notebook/data, the complete eight-page `reseach.pdf`, and the 2026 reconstruction runs. The related `pi-vae` folder contains 298 files, 267,073,655 bytes, including two sets of fitted neural/tree models, posterior draws, chain CSVs, predictions and plots. Neither adjacent folder is a Git checkout. Its present package is already the 2026 reconstruction; the ZIP and `legacy/` establish which five artifacts are historical.

All five original ZIP entries match their retained original counterparts byte for byte. ZIP timestamps are 1 October 2026: packaging dates, not evidence of when research or authorship occurred. All 14 source hashes recorded by the 500-epoch run match the supplied reconstruction package before cleanup. Earlier `SOURCE_AUDIT.md`, `REVIEW.md`, and `RESULTS.md` were treated as claims to check, not independent proof.

The publication is Imane Fakir, Mehdi Nejjar, Mohammad Hachim Eraissouni, Adam Kessab, and Ahmed Khallaayoun, “A Hybrid πVAE-XGBoost Framework for High-Fidelity Simulation of Bifacial Solar Thermal Collectors,” *Materials Research Proceedings* 64 (2026), 171–178, DOI [10.21741/9781644904091-21](https://doi.org/10.21741/9781644904091-21). The [publisher](https://mrforum.com/product/9781644904091-21/) gives online publication as 25 April 2026. Imane Fakir is the published first and corresponding author; Mehdi is second. The user's account that Mehdi invited him remains valid as a personal account, but its first-author wording should be corrected. The paper contains no individual contribution taxonomy.

## Historical source and upstream provenance

The five original artifacts are `train_custom.py` (18,169 bytes), `pivae_2d.stan` (1,771), `data_analysis/data_mu_std.ipynb` (1,065,681), and the two CSVs. There is no historical XGBoost implementation, MATLAB/Simulink model, CARNOT configuration, fitted neural model, retained raw MCMC file, experiment log, requirements file, or documented model search. The notebook references an unavailable `mcmc_1000.csv`.

Python/PyTorch implements an RBF feature map Φ, two tanh hidden layers, 64 basis coefficients, a coefficient encoder/decoder VAE with widths 128/64 and latent dimension 24, and reparameterized Gaussian sampling. Stan transfers fixed decoder weights and samples latent z and observation noise, using a standard-normal z prior and a Gaussian likelihood. The encoder output named `sd` is actually log variance in the reparameterization/loss; `sigma2` is used as a standard deviation in Stan.

Upstream comparison was made against [MLGlobalHealth/pi-vae](https://github.com/MLGlobalHealth/pi-vae), commit `14de0377500ce175115a6509a802e473c3466a54`, referenced by [Mishra et al. (2022)](https://doi.org/10.1007/s11222-022-10151-w). After Python AST formatting normalization, the historical Encoder, Decoder and VAE classes and the relevant πVAE loss function match exactly. PHI similarity is 0.995 and PIVAE similarity 0.903, including substantial matching documentation. This supports adaptation of that implementation, not independent invention of these architectures. It does not establish who made the adaptations. Upstream uses MIT, copyright 2022 Machine Learning and Global Health Network; its notice must accompany substantial reused portions. No corresponding notice was present in the supplied source.

Historical training uses nine features, omitting inlet temperature, and explicitly sets `target_col = 'm_flow_1kgh'` at line 291. Its RBF center count is half the test row count (8,640), centers start as random normal values, inputs are unscaled, batch size is 32, Adam rate is 0.0003, and planned training is 10,000 epochs. There are 32 trainable beta rows attached to shuffled minibatch slots rather than persistent observed functions. Loss sums two reconstruction SSEs and KL with no separate weight. The unused `alpha` argument does not control the hard-coded RBF factor. Large unscaled distances can saturate/underflow the RBF features; no retained trained historical model establishes the actual extent.

Every 200 epochs including epoch zero, Stan conditions on **all test labels**: lines 368–390 set `x_inf = val_ds.evalPoints`, `y_inf = val_ds.data`, and all test indices in the likelihood. Sampling uses four chains, 200 warmup and 1,000 retained draws per chain, adapt_delta 0.95; seeds are not specified. Test loss also selects `best-model.pth` at lines 467–472. Validation generally remains in training mode except at HMC epochs, so its VAE draws are stochastic. The source creates WandB logging and compiles Stan at import, then invokes training at line 481. It should not be run as a public reproduction entrypoint.

Metric code uses sklearn R² and scipy Pearson/Spearman correctly as arithmetic, but on test-conditioned predictions for the mass-flow target. Stan's `y2` adds observation noise: its bands are posterior predictive, despite historical “credible” labeling. Random 250-row display windows have no seed. Models/MCMC files named by the script are expected outputs, not retained evidence.

The notebook has 21 code cells, including an empty final cell and seven stored PNG figures. Its cells concatenate training and test, compute global Q mean/std, and apply global z-scores. The posterior-plot cells de-normalize Q using those combined statistics. Their outputs support a Q-oriented analysis variant; they do not connect the missing posterior to a particular training version. There is no metric table or tree fitting in the notebook, and no author/version history. Non-monotonic execution counters show interactive use, not a recoverable chronological experiment record. Partial execution through preprocessing is feasible; full execution fails at the missing MCMC file.

## Dataset and preprocessing

The paper attributes the measurements to the Institute of New Energy Systems (InES), Germany, supplied with consent for this research. There is no distribution agreement or dataset license in the inspected materials. Filenames containing “simulation” do not independently establish synthetic provenance; the paper describes measurements.

Both CSVs contain 22 columns: `T_in_1C`, `T_out_1C`, `T_mass_flow_1C`, `T_amb_outdoorC`, `T_vent_1C`, `Vol_flow_1lh`, `v_wind_speedms`, `G_trackerWm`, `G_diffuseWm`, `IR_TrackerWm`, `T_IR_Tracker`, `cp_1kJkgK`, `m_flow_1kgh`, `rho_1kgl`, `dT_1K`, `Q_1kW`, `SecondsSinceStart`, `Datum`, `zenith`, `azimuth`, `direct_rad_calc`, `diffuse_rad_calc`. The original acquisition, sensor calibration, cleaning, time-zone convention, and generation of derived radiation/material-property columns are absent.

Training has 130,048 rows from 21 August 2024 04:00:05 through 29 August 16:37:20, excluding 28 August entirely. Counts by day: 21: 14,399; 22–27: 17,280 each; 29: 11,969. Test has 17,280 rows on 28 August 00:00:00–23:59:55. Within recorded periods the spacing is five seconds. There are no shared or duplicate timestamps, missing/nonfinite numeric values, or unordered timestamps. Total: 147,328, **68 fewer** than the paper's 147,396. No rows were added, removed or interpolated.

Raw SHA-256: train `2c63259c638777b0c988ff4919847bc345ceb8489cefb7ac95e54dd73ee2433f`; test `437dfdeb2c9e02476cdfcaf2a8fafc768be5ab4a3abab099fc3903cde7b7f319`.

Numerical identities hold to CSV precision: `dT = T_out − T_in` within 1.00000001e-6 K, and `Q = m_flow * cp * dT / 3600` within 8.358354e-7 kW. Using dT/outlet to predict Q would encode the target; using measured Q with inlet/flow/heat capacity to predict outlet can nearly solve it algebraically. The reconstruction's enforced feature allowlist excludes measured Q, outlet and dT. Other omitted columns have uncertain inference-time availability; exclusion does not prove that each is intrinsically leaky.

## October 2026 reconstruction

The available final hybrid implementation is **PyTorch + Stan/CmdStanPy → scalar predicted Q → XGBoost outlet regression**. Ten inputs follow Fig. 3: inlet, ambient, vent and tracker temperatures; global, diffuse and infrared irradiance; wind, zenith and azimuth. Latent z/beta are not tree inputs. Input and Q standardization, center initialization and neural/HMC fitting use each training fold only. Four GroupKFold splits hold out whole days; every training row gets Q from a fit excluding its day. A separate full-training fit supplies test Q. Training includes 29 August, so this is held-out-day reconstruction, not future forecasting.

The 500-epoch profile uses 128 centers, Φ widths 128/128, beta 64, latent 24, VAE widths 128/64, 32 persistent coefficient replicas of the same observed Q function, batch 2048, Adam 0.0003, RBF factor 0.25, KL weight 0.001 and gradient clipping 10. CPU seeds are explicit, deterministic PyTorch operations enabled, with two threads. There is no validation-based epoch selection. These are reconstruction choices, not recovered final author settings.

Stan uses all fitting rows through an exact augmented-QR Gaussian likelihood, including the `−n log(sigma)` normalization term. It transfers the learned decoder and keeps neural weights fixed. The sigma prior numerically retains Normal(0.025,0.01), lower bound 0.001, but is now in standardized-Q units; its physical meaning therefore differs from the historical raw-target prior. Four chains each retain 1,000 samples after 1,000 warmup, dense metric, adapt_delta .95, max tree depth 12. Twelve training-only L-BFGS starts select a high-density initialization; small-jitter chains describe exploration near that mode, not global posterior mode weights. Optimizer initialization is not the point estimator.

XGBoost uses squared-error regression, histogram trees, 500 estimators, max depth 6, rate .05, subsample and column subsample .9, lambda 1, two jobs and the seed 20261003. Direct Q, direct outlet, πVAE-Q outlet and XGBoost-Q outlet comparisons exist. Stacks use out-of-fold Q for fitting; outlet training scores are nevertheless in-sample second-stage scores. No `eval_set` or test early stopping is used. The 80-epoch verification profile is a smaller **2026** configuration, not evidence of a historical architecture being tried/rejected.

`model.py`, `training.py`, `posterior.py` and the packaged Stan source implement the deep component; `stage_one.py` supplies Q fits/cross-fits; `stage_two.py` implements trees; `data.py` and `config.py` enforce boundaries; `inference.py` performs frozen prediction; `evaluation.py` creates metrics; `plots.py` creates loss, interval, scatter, outlet, full-record and importance PNG/SVG plots; `cli.py` orchestrates commands; `verification.py` checks target invariance. Configs/requirements define the environment. Existing tests check leakage boundaries, fold separation, QR equivalence, metrics, checkpoints and fitting inputs.

## Paper-to-source mapping

| Paper item | Supporting historical evidence | Reconstruction counterpart | Status |
|---|---|---|---|
| p.172, 147,396 five-second observations | CSVs cover the period with 147,328 rows | `data.py`; source inventory | Count discrepancy; acquisition/cleaning unrecovered |
| Eq. 1: integrated energy | No numerical energy-integration implementation | Target `Q_1kW` instantaneous power | Energy vs power/unit mismatch; no replication |
| Table 1 variable definitions | 18 named variables present; CSV also has time and derived-radiation columns | Ten-variable allowlist | Schema supported; physical calibration unknown |
| Figs. 1–2 Simulink/CARNOT structure | None | None | Diagram only; classical model absent |
| Fig. 3(a), πVAE predicts Q from ten inputs | RBF/coefficient-VAE/Stan topology; historical target is mass flow and inlet absent | `model.py`, `posterior.py`, `stage_one.py` | Structural correspondence; historical final-Q driver missing |
| Eqs. 2–4, decoder/basis/posterior mean | Original Python/Stan; posterior-noise draws saved by script in theory | `predict_posterior`: E[d(z)]·Φ(x) | Formula supported; historical posterior artifact absent |
| Fig. 3(b), Eq. 5, XGBoost outlet stage | No tree code or fitted tree | `stage_two.py`, `inference.py` | Added in 2026; not recovered historical implementation |
| p.175, whole held-out 28 August | CSV split matches | Whole-day cross-fitting and frozen test prediction | Historical source then conditions on test labels; clean held-out claim not supported by that source |
| Fig. 4 CARNOT vs measurements | No simulator outputs; axes show 2023 while methods/data use 2024 | No fabricated replacement | Unreproducible |
| Fig. 5(a), training/test loss through about 500 epochs | Original logging code, but no retained loss log; source plans 10,000 epochs | `plots.py`, `training_loss` | New 2026 training-only plot, not the paper figure |
| Fig. 5(b), tree importance | No historical tree/model/gain file | `stage_two.py`, `*_importance` | New 2026 results; gain is not causal explanation |
| Table 2: πVAE .9998/.9986/.9990; tree 1/.9999/1; hybrid .9999/.9992/.9995 (R²/Spearman/Pearson) | No score/prediction provenance; under “Training Results” with unclear tasks/splits | `evaluation.py`; saved 2026 metrics | Paper-reported only; p.177 still says a performance table is awaiting |
| Fig. 6(a–b), standardized-Q prediction and 95% bands | Notebook Q posterior plots and original Stan plotting are plausible sources; missing `mcmc_1000.csv`, exact match unestablished | `test_q_intervals`, `test_q_zoom` | Historical source uses predictive-noise bands; “confidence/credible” wording ambiguous |
| Fig. 7(a), Q across the complete period | Notebook data/Q plots, but no complete historical predictions | `full_record`, `test_q_intervals` | Exact historical figure unlinked; whole-record plot combines training fits and test predictions |
| Fig. 7(b), outlet across period | Notebook measured outlet only; no historical tree predictions | `test_outlet`, `full_record` | Exact historical figure unlinked |

Visual resemblance is not a cryptographic or numerical link to a published figure. Historical deep-learning figure work plausibly concerns Fig. 3(a), Fig. 5(a), Fig. 6 and Fig. 7(a), but individual creator attribution requires the user's account or corroborating author records. Fig. 5(b)/7(b) concern tree/outlet work and cannot be assigned personally from these files.

## Independently checked reconstruction results

All 19 existing tests pass. All 12 retained model/split metric records were independently recomputed with sklearn/scipy to differences below 1e-12. Frozen inference on all 17,280 test rows matches saved values after the same CSV serialization, including interval endpoints. Removing or poisoning Q/outlet/dT leaves all output columns bit-identical. All five retained posterior archives exactly match raw Stan z/f/sigma draws; decoder transfer differs by at most 5e-16. No training or replacement of historical results occurred.

| Model and task | R² | MAE | RMSE |
|---|---:|---:|---:|
| πVAE, Q in kW | .994684 | .042240 | .072010 |
| Direct XGBoost, Q in kW | .997169 | .031508 | .052557 |
| Direct XGBoost, outlet °C | .997849 | .106478 | .171045 |
| πVAE-Q → XGBoost, outlet °C | .997968 | .103431 | .166222 |
| XGBoost-Q → XGBoost, outlet °C | .997262 | .114499 | .192972 |

The hybrid's outlet RMSE gain over direct XGBoost is 0.004823 °C, 2.82%, on one day. πVAE is worse than direct XGBoost for Q. An explicitly exploratory review calculation, `T_out = T_in + k*predicted_Q` with k=3.079510965 fitted on training labels, gives RMSE .161861 °C using existing XGBoost Q, slightly better than the hybrid. This was review-driven and is not a pre-specified or historical benchmark. It challenges a necessity/superiority claim, not the arithmetic validity of the saved hybrid result.

Latent-function observed-Q coverage is 5.150%, width .002407 kW; observed-Q comparison is not a nominal observation calibration test for a latent interval. Observation predictive coverage is 94.473%, width .305646 kW, but falls to 42.64% at noon and 65% in the next hour. Residual lag-one correlations are .95824 (πVAE Q) and .98121 (hybrid outlet). Whole-day averaging and many strongly dependent five-second rows do not prove broad uncertainty calibration or significant model superiority.

All relevant retained Rhat/bulk/tail ESS values are finite. Across five fits max Rhat is 1.00523, min bulk ESS 2180.04, min tail ESS 1661.2; saved diagnostics report no divergences/depth saturation. The current threshold 1.05 is permissive compared with [Stan guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html); stricter diagnostics belong in a separately declared future experimental profile. Small-jitter starts near a selected mode limit global conclusions even when within-mode diagnostics are good.

## Contribution evidence and scientific limits

The paper establishes co-authorship and discusses the implemented deep method. The source establishes an upstream-derived implementation and project-specific data/training/Stan/analysis integration. Neither supplies individual authorship, original framework proposer, code-review decisions, abandoned architecture results, or reasons for historical revisions. Unused Gaussian-process/Matern imports are not evidence of a fitted comparator. Commented beta alternatives and the mass-flow/Q mismatch suggest exploratory development, but do not establish a dated model-selection narrative.

The user's account supplies his implementation/evaluation role, iterative critique, manuscript/figure work, invitation by Mehdi, and time constraints. Those statements should be labeled self-reported where evidence is discussed and may be corroborated by Dr. Khallaayoun. The 2026 reconstruction and this cleanup must not be represented as his unaided original research implementation. Future experiments need nested whole-day outlet validation, repeated seeds, fresh external data, a physical baseline, ablations, sensor-uncertainty context, temporal/heteroscedastic noise and mode-sensitivity analysis. Replicas trained on a single observed function differ from the foundational πVAE's collection of independent prior-function draws; training a map/decoder on Q and then conditioning on Q is a conditional, data-dependent approximation, not full Bayesian uncertainty over neural weights.

## Cleanup, security and publication disposition

Two byte-identical duplications exist: root `SOURCE_AUDIT.md` versus package `docs/SOURCE_AUDIT.md`, and root versus packaged current Stan source. Originals are retained privately; the public candidate will have one authoritative Stan source and consolidated documents. No dataset, notebook or weight is present in the requested 34-file code folder. Adjacent datasets, binary models/posteriors/compiled Stan executables and row-level plots/predictions remain private. The historical notebook's outputs contain measurements and will be stripped from a source-only historical copy.

A pattern scan of all supplied 34 files found no recognized credential/private-key patterns and seven personal-path occurrences in documents. No history exists to scan. This is a bounded check, not proof against every possible secret format. Historical WandB identifiers are logging labels, not embedded API credentials. Public documents will use relative paths and portable data arguments. Local inventories, full original archive, original notebook outputs, private run artifacts and absolute paths will stay outside the public candidate.

Two reconstruction reliability defects were independently confirmed from source/artifacts: frozen πVAE loads an unbound neural/posterior pair; twelve optimizer starts write into one timestamp-named directory, retaining only one raw optimizer CSV per fit. The cleanup may add hash binding, source-continuation checks and unique log directories without changing architecture, targets, likelihood, hyperparameters or recorded results. These changes are repository maintenance, not silent corrections to the historical experiment.

No root code license or ownership grant is available. Upstream MIT applies to its reused material; the article's CC BY 3.0 notice applies to the article and does not establish a separate solar-code or InES-data grant. Public publication remains on hold until the authorized project rights holder(s) or institutional authority confirm permission to redistribute the project adaptations and any approved derived artifacts. No dataset will be published without separate InES authorization. No blanket license should be selected from co-authorship alone. GitHub authentication is available; access is not the blocker.
