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

InES measurements have no established redistribution grant here. CSVs, raw predictions, weights/posteriors, notebook outputs and generated data-derived images remain private. New aggregate summaries are included at the user's direction; independent confirmation of their permitted use is absent from the recovered record.

## October 2026 maintenance and citation metadata

The reconstruction, tests, forensic review, artifact-binding corrections and application-language drafts are AI-assisted maintenance/reconstruction. They are distinct from the historical research contribution. Existing source records do not establish an exact individual reconstruction-author roster. `CITATION.cff` therefore uses a descriptive collective software-contributor label and a verified `preferred-citation` containing the article authors. That collective label is not an institution, an ownership claim, or an attribution of every code file to every paper author. Resolve the software contributor roster with the team before a definitive software citation/release.

## User-directed GitHub publication and unresolved licensing

Repository: [https://github.com/Mr-Espada/bifacial-solar-pivae-xgboost](https://github.com/Mr-Espada/bifacial-solar-pivae-xgboost); initial source release `v0.1.1`. Following local preparation and notice of the unresolved rights questions, Mohammad explicitly instructed: “well you should publish it in my github” and “create the repo and publish the work.” The repository is therefore uploaded publicly at his direction.

This is evidence of the user's upload instruction, not independent evidence of consent from every co-author, an institution or the data provider. No blanket code license has been selected, and no third-party permission has been invented. The ownership allocation, an agreed project license, institutional/project requirements if applicable, and permission for new aggregate summaries remain unverified. Raw data and fitted/row-level artifacts remain excluded.

The [licensing note](../LICENSE.md) and [publication record](RELEASE_CHECKLIST.md) preserve these distinctions. The earlier pre-cleanup audit describes the hold that existed when it was written; it is retained as a dated snapshot rather than rewritten to imply that the permissions were independently verified.
