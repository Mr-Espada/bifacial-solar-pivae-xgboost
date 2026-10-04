# Validation record — 4 October 2026

All original files and raw scientific artifacts were read-only. Validation ran in isolated working/run/install locations. No full neural/HMC training, original-result replacement or GitHub upload occurred.

| Check | Observed result | Boundary |
|---|---|---|
| Original requested folder | 34 files inventoried/hashed; pre-cleanup archive reopened and all members checked | No original file modified/deleted/moved |
| Historical ZIP | All five original artifacts match retained copies | Archive timestamps do not establish research dates/authorship |
| Saved reconstruction source | All 14 source hashes match supplied pre-cleanup package | No historical Git lineage exists |
| Original test suite | 19 passed | Synthetic/unit checks, separate from empirical result validation |
| Final maintained test suite | 36 passed in compatible environment | Includes log-collision, failed/nonfinite quality gates, model/posterior/diagnostic swapping, hybrid-stage provenance, source drift and interrupted-stage log protection |
| Installed package | 0.1.1 wheel built and installed to isolated target; 13 modules import; command help works | Uses retained dependencies plus separate SymPy 1.13.1 overlay; no fresh whole-toolchain installation |
| Dependency consistency | `pip check`: no broken requirements | Old SymPy 1.13.3 mismatch documented and left unchanged in original runtime |
| Wheel contents | Packaged Stan source and full upstream MIT notice verified | No blanket project code license selected |
| Data audit command | Both expected original SHA-256 values match | No data bundled or redistributed |
| Independent score recomputation | All 12 saved model/split records match within 1e-12 | These are 2026 reconstruction records, not published Table 2 |
| Frozen inference | All 17,280 test rows/columns match saved predictions after same CSV serialization, both original source and installed maintained package | No fitting; float serialization precision distinguished from in-memory arrays |
| Target independence | All output columns bit-identical after target deletion/poisoning on all test rows | Tests input-target dependence, not proof of every possible modelling assumption |
| Posterior archives | All five beta/z/sigma exports exactly match raw chain CSVs; decoder error below 5e-16 | Within-mode diagnostic quality does not prove global exploration |
| Saved-run import | Explicit new private copy; model/CSV bindings added; original source run hash map unchanged | New hashes are post hoc, not historical authentication; lost logs remain lost |
| Finalized saved-run import | New private copy with diagnostic hashes and hybrid-stage binding; all 17,280 serialized outputs still match | Installed final wheel; no fitting or original-run edits |
| Finalized evaluation/invariance | All 12 records and complete `metrics.json` unchanged; ten PNG/SVG pairs regenerated; all-row absent/poisoned-target invariance passes | Same saved models; plots remain private; no new paper metrics |
| New target-free input command | CLI predicts 128 target-free rows successfully | A bounded command check, separate from the complete held-out replay |
| Maintained evaluation | `evaluate` regenerated metrics and ten PNG/SVG plot pairs; complete `metrics.json` matches original | Generated data-derived graphics remain private; representative full-day Q/outlet plots visually inspected |
| Historical notebook | First nine original cells executed; missing MCMC file stops cell 9 | Full execution/reproduction unavailable; all 21 code-cell sources preserved in stripped public copy |
| Citation | Official CFF 1.2.0 JSON Schema validation passes; five publication authors preserved in preferred citation | Collective software label is explicitly provisional; individual authorship unresolved |
| Release scan | Candidate text/files checked for recognized secrets, personal paths, large/binary/data payloads and notebook outputs | Bounded pattern scan, not universal secret proof; original history absent |
| Public-approval gate | Deliberately fails for two unresolved rights-holder authorizations | This is the user's requested ownership boundary; no public remote/repository created |

Verified commands from the README include tests, audit, explicit saved-run import, predict/inference, evaluate, verify-run, command help and release scanning. New long-profile training/Stan installation commands are documented but were not re-executed. A working pinned Stan toolchain was read for retained posterior checks; complete new training and alternate platforms remain unverified.

Machine-readable non-sensitive validation is in `VALIDATION.json`. Detailed hash inventories, original archive, independent recomputation, notebook and wheel/citation validation records remain in the private audit/history directory alongside the prepared project. `RESEARCH_AUDIT.md` records findings before candidate modifications; `CHANGELOG.md` records the separate maintenance actions.

The finalized candidate contains 56 source/document files. All 335 original fingerprints and timestamps were rechecked unchanged. The earlier 52-file candidate was also snapshotted and its archive members verified before finalization. Scientific core files (`model.py`, `training.py`, `config.py`, packaged Stan) and both experiment configs remain byte-identical to the supplied reconstruction. The source-only historical notebook still contains all 21 original code-cell sources with no outputs.

The final wheel's SHA-256 is `c45f5978225f7dc2fb2b05883d922732ba22fa61fc0f7610d5b3db9772603d9e`. Its isolated installation, all 13 module imports, dependency consistency, packaged Stan/upstream notice, CLI/script help, full-day imported inference and metric/plot paths were checked again. The citation passes the official CFF schema with Mehdi's profile link; local Markdown references have no broken destinations.

The final local repository has a new maintenance commit, no inherited research commits and no remote. Every staged blob was compared with the scanned candidate tree before committing. This commit records October 2026 preparation and contributor acknowledgement; it must not be described as original research history or a public release. The final SHA, source-only ZIP hash and local receipt are recorded outside the candidate to avoid a self-referential commit record.
