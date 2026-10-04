# Contributing

Open an issue or pull request with a brief description of the change. For code changes, run:

```bash
python -m pytest -q
python scripts/check_release.py
```

Keep data, model weights, posterior samples, row-level predictions and credentials out of Git. Use synthetic fixtures in tests.

For new experiments, save the configuration and diagnostics in a fresh run directory and add a dated changelog entry. Keep earlier results intact, and label new comparisons separately from the published study.

Retain notices and citations when adapting third-party code.
