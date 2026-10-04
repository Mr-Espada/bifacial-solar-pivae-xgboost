# Attribution and publication rights

## Research publication

Imane Fakir, Mehdi Nejjar, Mohammad Hachim Eraissouni, Adam Kessab, Ahmed Khallaayoun. “A Hybrid πVAE-XGBoost Framework for High-Fidelity Simulation of Bifacial Solar Thermal Collectors.” *Materials Research Proceedings* 64 (2026): 171–178. [DOI](https://doi.org/10.21741/9781644904091-21). The [publisher](https://mrforum.com/product/9781644904091-21/) lists online publication on 25 April 2026 and CC BY 3.0 for the article. Preserve all five authors and their published order.

Article authorship is not evidence that each author wrote every file, nor a grant to license jointly or institutionally owned project code. There is no individual contribution statement in the article. Mohammad's role statement is explicitly based on his account; see `CONTRIBUTION_EVIDENCE.md`.

## Named historical contributors

At Mohammad's explicit direction on 4 October 2026, **Mehdi Nejjar** ([Night^^Stalker / @mehdinejjar86](https://github.com/mehdinejjar86)) is credited for suggesting the overall research framework and contributing to project code. Mohammad is credited primarily for implementing and critically evaluating the deep-learning component, model revisions, its manuscript section, results and figures. He did not originate the framework. These roles are recorded in [CONTRIBUTORS.md](../CONTRIBUTORS.md); they do not independently identify every file author or authorize release. The GitHub profile/handle is verified; the role allocation comes from Mohammad's account.

## Upstream πVAE

The historical solar script substantially adapts [MLGlobalHealth/pi-vae](https://github.com/MLGlobalHealth/pi-vae). Audited upstream commit: `14de0377500ce175115a6509a802e473c3466a54`. Encoder, Decoder, VAE and the relevant coefficient-loss code match after AST formatting normalization; PHI and PIVAE have substantial correspondence. This is source evidence of reuse, not proof of which solar-project author adapted it. The October reconstruction inherits/refactors that lineage.

Preserved upstream MIT notice: [third_party/pi-vae/LICENSE](../third_party/pi-vae/LICENSE), copyright (c) 2022 Machine Learning and Global Health Network. It is **only the upstream notice**, not a blanket license of all solar-project modifications. Upstream source is not vendored wholesale; the checked comparison and hashes are recorded privately.

Cite the method: S. Mishra, S. Flaxman, T. Berah, H. Zhu, M. Pakkanen, S. Bhatt. “πVAE: A stochastic process prior for Bayesian deep learning with MCMC.” *Statistics and Computing* 32, 96 (2022). [DOI](https://doi.org/10.1007/s11222-022-10151-w).

## Third-party software and data

Dependencies are separate software distributions: PyTorch; Stan/CmdStan/CmdStanPy; XGBoost; NumPy, pandas, SciPy, scikit-learn, matplotlib and pytest. This repository does not grant rights to them or include their full source/binaries. Preserve their own licenses if redistributing dependencies. XGBoost's methodological reference is T. Chen and C. Guestrin, [“XGBoost: A Scalable Tree Boosting System”](https://arxiv.org/abs/1603.02754), 2016.

InES measurements have no established redistribution grant here. CSVs, raw predictions, weights/posteriors, notebook outputs and generated data-derived images remain private; new aggregate summaries also require confirmation of their permitted use.

## October 2026 maintenance and citation metadata

The reconstruction, tests, forensic review, artifact-binding corrections and application-language drafts are AI-assisted maintenance/reconstruction. They are distinct from the historical research contribution. Existing source records do not establish an exact individual reconstruction-author roster. `CITATION.cff` therefore uses a descriptive collective software-contributor label and a verified `preferred-citation` containing the article authors. That collective label is not an institution, an ownership claim, or an attribution of every code file to every paper author. Resolve the software contributor roster with the team before a definitive software citation/release.

## Public-release hold

No root `LICENSE` has been selected. No remote repository was created or uploaded. The technical candidate is prepared locally; public release awaits confirmation from the authorized rights holder(s):

1. Permission to publish the solar-project adaptations and the historical source excerpts, including any co-author/university ownership requirements and contributor acknowledgement.
2. Permission under the InES/data agreement to share the new aggregated reconstruction summaries. Actual data/binary/row-level artifacts remain excluded and would need a separate distribution grant.
3. Agreement on any project-wide code license if one is desired. Upstream MIT notices must remain intact regardless; article CC BY does not substitute for this grant.

The explicit [licensing note](../LICENSE.md) and [release requirements](RELEASE_CHECKLIST.md) identify the remaining permissions. This hold follows both the original Phase 6/8 instruction and the follow-up's licensing/publication requirement. Authentication/access to GitHub is available. A public URL is absent because publication has stopped at the ownership boundary after local preparation.
