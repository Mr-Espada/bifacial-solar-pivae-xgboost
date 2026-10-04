# Changelog

## 4 October 2026 — documentation update

- Shortened the README, credits, licensing and supporting notes.
- Moved the application contribution sheet and earlier development reports to the local archive.
- Updated citation metadata. Code, experiment settings and aggregate results are unchanged.

## 0.1.1 — 4 October 2026

- Published the source repository with the paper citation and upstream πVAE MIT notice.
- Added hashes linking models, posteriors, diagnostics and predictions. The hybrid tree is linked to the πVAE stage used to generate its inputs.
- Made failed or missing posterior diagnostics stop downstream training and inference.
- Added separate optimizer logs and protection against overwriting interrupted fitting stages.
- Added a saved-run import tool and regression checks for artifact mismatches, source changes and fitting boundaries.
- Pinned the compatible SymPy version and checked the installed package, saved inference, metrics and plots.
- Retained historical Python/Stan source and notebook code. Data, notebook outputs, models and row-level outputs remain outside Git.

These maintenance changes affect checks and failure handling. They do not replace historical results or change the reconstruction's architecture and numerical settings. Reconstruction and repository maintenance used AI assistance.

## 0.1.0 — 3 October 2026

Reconstructed the πVAE-Q → XGBoost-outlet pipeline with training-only preprocessing, whole-day cross-fitting, direct-tree baselines, fixed seeds and separate training, prediction and evaluation commands. Saved results are reported under this reconstruction's settings, rather than as a reproduction of the paper's Table 2.

## Historical study

The paper was published on 25 April 2026. Retained source consists of a PyTorch/Stan script, Stan model, analysis notebook and two measurement files. The original final hybrid models and complete experiment settings are unavailable. Original files and outputs are preserved locally.
