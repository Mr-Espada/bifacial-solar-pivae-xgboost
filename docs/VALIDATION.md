# Validation

Checks completed on 4 October 2026 used isolated working directories and preserved the original files. No full neural/HMC retraining was performed.

| Check | Result |
|---|---|
| Original files | All 335 original fingerprints and timestamps remained unchanged; original archives were reopened and verified. |
| Tests | 36 passed, including leakage boundaries, QR likelihood, artifact mismatches, posterior quality and log protection. |
| Package | Version 0.1.1 wheel built and installed; all 13 modules imported and command help worked. |
| Dependencies | `pip check` passed with the compatible SymPy 1.13.1 overlay. |
| Package contents | Stan source and the upstream MIT notice were included. |
| Data | Both original CSV hashes matched. Data files remain outside Git. |
| Metrics | All 12 saved model/split records matched independent recalculation within 1e-12. |
| Frozen inference | All 17,280 test rows, including interval endpoints, matched saved predictions after the same CSV serialization. |
| Target independence | Removing or poisoning target columns left every test prediction unchanged. |
| Posterior exports | All five z/beta/sigma archives matched raw Stan CSVs; decoder transfer error was below 5e-16. |
| Saved-run import | A separate copy received artifact bindings and retained identical serialized predictions. |
| Evaluation and figures | Metrics remained unchanged; ten PNG/SVG pairs were generated locally. |
| New-input prediction | The command successfully predicted 128 rows without target columns. |
| Historical notebook | The first nine cells ran; execution then stopped at missing `mcmc_1000.csv`. All 21 code cells remain in the source-only copy. |
| Citation and source scan | CFF schema and local links passed; scans checked for recognized secrets, personal paths, data, binary artifacts, notebook outputs and oversized files. |

The installed wheel's SHA-256 was `c45f5978225f7dc2fb2b05883d922732ba22fa61fc0f7610d5b3db9772603d9e`. Machine-readable results are in [VALIDATION.json](VALIDATION.json). Detailed execution records and original artifacts are archived locally.

The documentation update changes prose and citation metadata only. Code, tests, configurations, the historical notebook and aggregate metrics remain unchanged. Documentation links, citation schema and release scans are checked again before pushing.

A complete new training run, fresh toolchain installation and cross-platform execution remain unverified. The paper's Table 2 and exact figures are not reproduced. Source scans check known patterns; HMC convergence checks do not establish predictive calibration.
