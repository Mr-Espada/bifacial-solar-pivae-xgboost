# Requirements before public GitHub publication

The source candidate is prepared privately. Proposed repository name: **bifacial-solar-pivae-xgboost**. No public repository or release tag exists yet.

## Outstanding authorization

1. **Solar-project code:** obtain confirmation from the actual code rights holder(s) that the solar-project adaptations, the selected historical Python/Stan source and the source-only notebook may be published. The confirmation should identify any jointly owned contributions and the license approved for the resulting repository. Mohammad and Mehdi are credited code contributors, but the recovered record does not establish the complete ownership allocation.
2. **Project/institutional rights:** ask Dr. Ahmed Khallaayoun to confirm, with the relevant Al Akhawayn University authority if applicable, whether a supervision, employment, funding or project agreement controls release. If it does, obtain the required institutional/project approval. Do not presume either that the university owns the code or that it has waived rights.
3. **Data-derived summaries:** obtain confirmation from the InES data provider/agreement holder that the new aggregate scores and descriptive summaries may be made public. The existing research-use consent does not document this permission. Raw data, weights, posteriors, row-level predictions and generated data-derived figures remain excluded; distributing them would need a separate, explicit grant.
4. **Software citation:** confirm the reconstruction contributor roster with the team. The paper's five-author citation is verified; the software citation currently uses an explicitly provisional collective name. Confirming historical roles alone does not identify every reconstruction author.

Keep the actual confirmations privately; record only their approved scope and appropriate non-sensitive references in `PUBLICATION_STATUS.json`. Set authorization flags only after those confirmations. Select a conventional project license only when the rights holders approve it, preserving the upstream MIT notice.

## Technical preparation completed locally

Historical/source preservation, artifact hashes, posterior quality gates, hybrid-stage binding, distinct optimizer logs, interrupted-stage protection, portable instructions, contributor credits, dependency pins, tests, bounded saved-run execution, notebook boundaries, citation validation and source scans are documented in `VALIDATION.md`. No full model retraining is required merely to publish the reviewed source.

## Publication after authorization

- Recheck the complete Git tree and staged changes; run both release scans and the tests. Inspect the reviewed publication-status and license changes.
- Create the GitHub repository in the requested account under the professional name above, then push the reviewed maintenance commit. Keep the initial commit's actual preparation date; do not import invented historical commits.
- Set the repository URL in the README/citation/status, commit that metadata, recheck the final commit, and create an initial version tag for that final commit if appropriate. Record its full SHA and tag in the final report.
- Verify the remote tree and release contents after upload. Do not call a local commit a public GitHub release.

This checklist follows the user's explicit instruction to stop public publication when ownership or data permissions are unresolved. It is not a new consent requirement inferred from a repository skill.
