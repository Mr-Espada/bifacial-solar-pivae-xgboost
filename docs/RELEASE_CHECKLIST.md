# Release checklist

Before publishing a source update:

- Check the changed files and citation metadata.
- Run tests for code changes and `python scripts/check_release.py --require-public-approval`.
- Exclude measurements, fitted artifacts, row-level outputs, notebook outputs and credentials.
- Retain third-party notices and review any licensing changes with the relevant rights holders.
- Verify the remote commit after pushing. Create release tags without replacing earlier tags.

The repository is [bifacial-solar-pivae-xgboost](https://github.com/Mr-Espada/bifacial-solar-pivae-xgboost); the initial source release is `v0.1.1`. Current publication metadata is in [PUBLICATION_STATUS.json](PUBLICATION_STATUS.json). No project-wide code license has been selected.
