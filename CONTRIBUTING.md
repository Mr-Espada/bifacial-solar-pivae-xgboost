# Contributing

This is an October 2026 reconstruction associated with a published study. Preserve the separation between historical source, reconstructed experiments and new diagnostics.

1. Describe whether a change is maintenance or a new scientific experiment. Record scientific changes in `CHANGELOG.md`, use a new run directory, and keep the configuration, source hashes and diagnostic reports.
2. Retain historical code and reported results as evidence. Add a new comparison rather than overwriting old outputs or changing the paper's claims.
3. Run `python -m pytest -q` and `python scripts/check_release.py`. Model/posterior/diagnostic bindings, whole-day fold boundaries and the hybrid's πVAE-stage binding must pass. Failed posterior diagnostics stop downstream use.
4. Supply synthetic fixtures for tests. Keep measurement files, trained artifacts, row-level predictions, original notebook outputs, generated measurement-derived figures and credentials out of ordinary Git.
5. Credit adapted code with its original notice and citation. Describe actual individual contributions; do not manufacture historical Git activity or infer file authorship from the paper author list.

The publication instruction remains separate from technical validation and licensing. `python scripts/check_release.py --require-public-approval` checks the explicit recorded instruction to upload publicly and prevents an unsupported blanket-license assertion. Do not treat it as ownership or data-provider permission; see `docs/RELEASE_CHECKLIST.md`.
