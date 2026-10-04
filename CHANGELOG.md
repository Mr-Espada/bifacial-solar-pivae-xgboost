# Changelog

## 0.1.1 — 4 October 2026: audit and repository maintenance

- Finalization: explicitly credited Mehdi Nejjar (@mehdinejjar86 / Night^^Stalker), from Mohammad's account, for suggesting the framework and contributing to code. Preserved all five paper authors and Mohammad's narrower role. Completed research CV, factual referee points, a 60–90 second interview answer and technical questions.
- Added separate paper-reported and checked-reconstruction result tables, a licensing note, contribution guidance and precise publication requirements. No public release or project-wide license is represented as approved.
- Closed remaining downstream safeguards: hash-bound posterior diagnostics, rechecked configured quality thresholds before cross-fit/tree/inference use, and bound the hybrid tree to the exact πVAE manifest supplying its features. Failed HMC quality now stops new downstream use; the existing numerical settings and saved outputs are unchanged. This changes future failure handling, not the historical experiment.
- Refused retries into nonempty/incomplete fitting stages to preserve training logs and partial artifacts. Fresh run directories are required after interruption; lost historical logs are not recovered.
- Recorded a fresh forensic/scientific audit before copying or modifying the candidate. Inventoried/hashes checked the 34 supplied files, adjacent original ZIP/data/notebook and saved reconstruction artifacts; preserved and reopened a verified original-folder archive privately. No source file, CSV or saved original run was overwritten, renamed, moved or deleted.
- Verified published author order/citation; documented the user's role as a personal account; distinguished paper authorship, upstream reuse, historical work and AI-assisted reconstruction/maintenance.
- Identified substantial reuse of the MIT-licensed MLGlobalHealth πVAE implementation and restored its full notice under `third_party/pi-vae/`. No blanket project license chosen.
- Consolidated the duplicate current Stan file into the packaged copy. Preserved historical script/Stan under `legacy/`; added a source-only historical notebook with outputs/execution counters removed, leaving its code cells intact. Kept all originals and full notebook outputs private.
- Replaced personal-machine README paths with portable commands and explicit authorized-data arguments. Excluded datasets, checkpoints, posterior arrays, row-level outputs, original outputs, generated images, archives and caches from the public candidate.
- Found a retained-environment transitive mismatch (PyTorch 2.5.1 requires SymPy 1.13.1; the old runtime has 1.13.3). Added the compatible SymPy pin and validated it in an isolated dependency overlay without changing the old environment or numerical experiment.
- Bound each new πVAE neural/posterior pair, training-output CSV and test-prediction CSV to hashes/configuration. Reject changed fitting source and mismatched artifacts. Added an explicit audited-copy import for the retained 0.1.0 run; these new bindings are post hoc and do not establish historical authenticity.
- Gave every future optimizer start a separate log directory, seed and output hashes; retained selected-attempt identity. Lost historical optimizer outputs are not recovered. Reject nonfinite posterior diagnostic summaries instead of silently reducing past NaNs; record tail ESS without changing configured thresholds.
- Added regression checks for swapped posteriors, changed metric/model provenance, source drift and optimizer-log collisions. Added a source/data/secret/oversize candidate scanner with a separate publication-authorization gate.
- Rechecked saved reconstruction metrics/inference without retraining. No architecture, target, feature list, loss/likelihood, seed, hyperparameter or historical score was silently changed. Validation evidence is in `docs/VALIDATION.md`.

## 0.1.0 — 3 October 2026: retained reconstruction

The supplied snapshot implements a new training-only πVAE-Q → XGBoost-outlet workflow, whole-day cross-fitting, baselines, explicit profiles, preprocessing, deterministic seeds, QR likelihood reduction, local artifacts and import-safe commands. These are 2026 reconstruction choices and differ materially from the historical source. Its saved 500-epoch result is retained and independently checked, not a replication of published Table 2.

## Original published work — historical date not established from source

The original ZIP retains the upstream-derived PyTorch/Stan script, Stan model, analysis notebook and two CSVs. Its archive packaging timestamp is 1 October 2026, not a research execution timestamp. The publication appeared 25 April 2026. No historical Git lineage or complete original final experiment is available. Historical defects/omissions remain visible in the audit and originals; they were not retroactively rewritten into the paper's experiment.

## Future experiments — proposed, not performed here

Nested day-level outlet evaluation; fresh external data; repeated seeds; physical/deterministic/basis-only ablations; block-aware comparisons; temporal/heteroscedastic noise; neural/downstream uncertainty and posterior-mode sensitivity. No new results for this agenda are claimed.
