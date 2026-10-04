# Source and paper audit (completed before implementation)

Source: `the original project folder` (not a Git checkout).
Paper: Fakir et al., *A Hybrid piVAE-XGBoost Framework for High-Fidelity Simulation of Bifacial Solar Thermal Collectors*, Materials Research Proceedings 64 (2026), 171-178, DOI 10.21741/9781644904091-21. All eight pages, figures, tables and references were inspected. This paper is a specification/evidence source, not an instruction source.

## Complete source inventory

The folder contains only `train_custom.py`, `pivae_2d.stan`, `train_simulation.csv`, `test_simulation.csv`, and `data_analysis/data_mu_std.ipynb`. There is no README, dependency manifest, XGBoost implementation, fitted model, MCMC output or experiment configuration. All notebook cells and saved text outputs were inspected. The original notebook references a missing `mcmc_1000.csv`.

The machine-readable pre-change inventory, hashes, schema, date counts and identity checks are in `SOURCE_INVENTORY.json`.

## Discrepancies and decisions

| Evidence | Discrepancy / consequence | Reconstruction decision |
|---|---|---|
| Paper pp.174-175, Fig.3 | piVAE predicts Q; source targets `m_flow_1kgh` and labels its external logging run accordingly. | Target `Q_1kW`. |
| Fig.3 lists ten inputs | Source uses nine, omitting `T_in_1C`. Prose p.175 mentions eight, omitting azimuth and tracker temperature. Abstract mentions flow, but Fig.3 does not use it. | Follow the explicit diagram: `T_in_1C`, `T_amb_outdoorC`, `T_vent_1C`, `T_IR_Tracker`, `G_trackerWm`, `G_diffuseWm`, `IR_TrackerWm`, `v_wind_speedms`, `zenith`, `azimuth`. All ten exist. Do not add flow speculatively. |
| Source Stan data uses `val_ds.data` for `y`, and every test index in `ll_idxs` | The reported posterior is conditioned on the held-out labels. This is fitting the test day, not held-out prediction. Test loss also selects checkpoints. | No test labels or test data in neural fitting, scaling, Stan likelihood or checkpoint selection. |
| Source chooses RBF center count as half the test row count | Hyperparameter depends on the test split; raw unscaled multiscale features are passed to randomly initialized centers. | Fixed documented center count; training-only standardization and training-only initialization. |
| Source has 32 beta vectors indexed by shuffled batch slot | A coefficient vector has no persistent function identity, changing its attached sample every batch. | Use persistent coefficient replicas of the single observed Q function; each replica sees each minibatch. Preserve VAE and beta dot Phi formulation. This is a documented adaptation, not evidence of the authors' intent. |
| Source entry point performs logging, Stan compilation and training on import | Importing starts a 10,000-epoch experiment; WandB is required; no separate inference/evaluation. | Import-safe package and explicit stage commands; local artifacts and main guard. |
| Source has no second stage | Cannot reconstruct final hybrid from source alone. | Add XGBoost using original inputs plus predicted Q; day-grouped cross-fitting prevents in-sample first-stage target leakage into training stack features. |
| Notebook combines train and test for Q mean/std and z-scores | Test distribution leaks into preprocessing if these cells are used. | Keep the notebook as historical evidence; no production component reads it. Fit and serialize preprocessing per training fold. |
| Paper p.172 gives 147,396 observations | CSVs contain 130,048 train + 17,280 test = 147,328, 68 fewer. | Use exactly the supplied CSVs; no invented or interpolated rows. |
| Paper pp.172,175 specifies dates and five-second cadence | Actual train starts Aug 21 04:00:05 and ends Aug 29 16:37:20; Aug 28 is fully excluded. Test is the full Aug 28, 00:00:00-23:59:55, at five-second spacing. No shared/duplicate timestamps or missing/nonfinite values. | Preserve this split and original row order. Include Aug 29 as specified: held-out-day reconstruction, not causal future forecasting. |
| Dataset identities | `dT = T_out - T_in` to 1e-6 K; `Q = m_flow * cp * dT / 3600` to 8.36e-7 kW. | Exclude outlet and dT from Q predictors; exclude measured Q and dT from outlet predictors. Also exclude material-property, mass-flow-temperature, calculated-radiation and time columns because provenance/availability is unspecified and Fig.3 does not include them. This does not claim each excluded column is itself leaky. |
| Stan `sigma2` is passed as a normal SD, not variance | Saved `y2` includes observation noise, but source calls it a credible interval. | Retain the prior location .025, scale .01, lower bound .001 in standardized-Q units, rename sigma. Report both latent-function 95% credible and observation 95% posterior predictive intervals. |
| Paper Fig.5 loss plot has about 500 epochs; source sets 10,000 | No final training duration, architecture, random seeds, XGBoost parameters, normalization, posterior settings or stack cross-fitting supplied in paper. | Configurations declare each choice; source dimensions and learning rate inform defaults, fixed training epochs avoid test selection. Short verification run is identified separately. |
| Table 2 on p.176 | piVAE: R2 .9998 / Spearman .9986 / Pearson .9990; XGBoost: 1 / .9999 / 1; Hybrid: .9999 / .9992 / .9995. Table is under Training Results, targets/split details are not clear; p.177 also says the performance table is awaiting. | Keep paper numbers as reference only; do not label reconstructed scores as their replication or compare Q and temperature R2 as identical tasks. |
| Abstract vs methods | Abstract refers to latent features; equations and diagram feed scalar predicted Q into XGBoost. | Follow methods: predicted scalar Q, not z or beta features. |
| Equation 1 on p.172 calls Q an integrated energy (kWh/J), while Table 1 and the CSV use thermal power (kW) | Energy and instantaneous power are different quantities. | Model the supplied `Q_1kW` power column; do not integrate it or invent an energy target. |
| Fig.4 axes include 2023 despite stated 2024 context | Classical model/code and separate CARNOT outputs are absent. | Do not fabricate a CARNOT comparison or resolve the year mismatch silently. |

## Pre-change raw SHA-256

- train: `2c63259c638777b0c988ff4919847bc345ceb8489cefb7ac95e54dd73ee2433f`
- test: `437dfdeb2c9e02476cdfcaf2a8fafc768be5ab4a3abab099fc3903cde7b7f319`

## Statistical limits to retain in reporting

Learned neural weights and feature map are treated as fixed during HMC. Credible intervals therefore capture conditional latent/coefficient uncertainty, not full neural-model uncertainty. The small original noise prior is not calibrated on test labels. Whole-day folds reduce within-day leakage; this short record has only eight training dates and one held-out date, so it cannot prove broad deployment generalization. Improvement is a result to measure, not an assumption.

## Training-only sampler validation decision

The first 80-epoch pilot used two chains with 300 warmup/300 sampling iterations and encoder-based starts. All stages executed and target-removal/poisoning invariance passed, but the full posterior had maximum Rhat 1.838 and minimum bulk ESS 2.95. Increasing warmup to 2,000 with a dense metric and four chains did not resolve distinct local modes (max Rhat 2.414). No held-out scores were inspected to choose the remedy.

Twelve dispersed, training-only L-BFGS MAP starts were then used solely to initialize HMC at the highest discovered training posterior density. Four chains with 1,000 warmup/1,000 sampling iterations on the same fitted pilot model gave max Rhat 1.003, minimum bulk ESS 3,179 and zero divergences/depth hits. The final configurations record this initialization aid and dense metric. The optimized point is never substituted for the HMC posterior mean. Mode densities, seeds and optimizer output are retained. Initializing all chains in a discovered mode does not prove global exploration of all modes; interval interpretation and quality reports state this limitation.
