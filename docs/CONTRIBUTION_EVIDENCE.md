# Contribution evidence sheet

Prepared 4 October 2026 for funded master's applications. Subject: Mohammad Hachim Eraissouni. This sheet distinguishes personal account from technical and publication evidence; it is not an independent authorship certification.

## What is established

**Publication:** Mohammad is the third listed co-author of [Fakir et al., Materials Research Proceedings 64 (2026), 171–178](https://doi.org/10.21741/9781644904091-21). Published order: Imane Fakir, Mehdi Nejjar, Mohammad Hachim Eraissouni, Adam Kessab, Ahmed Khallaayoun. Imane, not Mehdi, is first/corresponding author in the final publication. State that Mehdi invited Mohammad, without calling Mehdi the published first author.

**Repository:** Retained historical Python implements a PyTorch RBF feature map, coefficient VAE, and fixed-decoder Stan/HMC inference. Its neural classes substantially reuse the MIT-licensed MLGlobalHealth πVAE implementation. Project adaptations include CSV loading, chosen input/target settings, neural training, Stan weight transfer, metrics/logging and plots. The notebook contains Q normalization/posterior plots. These artifacts establish technical work, not which individual wrote each part. There is no local Git history, notebook authorship metadata, historical XGBoost source, final-Q checkpoint or manuscript revision history.

**Personal account:** Mohammad says he was invited by Mehdi, was responsible principally for implementing and critically evaluating the deep-learning component, translated proposed ideas into experiments, helped prompt revisions, wrote/contributed to the deep-learning manuscript section, and prepared/contributed to results and figures. He explicitly credits **Mehdi Nejjar** ([Night^^Stalker / @mehdinejjar86](https://github.com/mehdinejjar86)) for suggesting the overall framework and contributing to project code. Mohammad did not originate that framework and believes time constraints limited experimentation. These role and decision-history statements come from him; the paper/source are technically consistent with the broad roles but do not independently prove their allocation. Exact code ownership and reconstruction authorship remain separate questions.

**Inference:** The mass-flow script versus Q notebook/paper suggests multiple stages/variants. It does not prove a dated decision sequence, who rejected an architecture, or why. Imported Gaussian-process/Matern names and commented coefficient alternatives do not establish fitted/evaluated competing models. The current XGBoost baselines, cross-fitting, 80/500-epoch profiles and cleanup tests are October 2026 work, not historical model-selection evidence.

## Academic CV: exact safe wording

- Co-author, “A Hybrid πVAE-XGBoost Framework for High-Fidelity Simulation of Bifacial Solar Thermal Collectors,” *Materials Research Proceedings*, 64, 171–178 (2026), DOI: 10.21741/9781644904091-21.
- Implemented and critically evaluated the project's deep-learning component using a πVAE-based PyTorch/Stan workflow, adapting an established implementation and contributing to iterative modelling revisions.
- Contributed to the deep-learning manuscript section and the preparation of experimental results and figures.

The second and third bullets rely on the personal account and should be corroborated by a supervisor where possible. Do not use “invented πVAE,” “designed the entire hybrid framework,” “independently developed all neural architectures,” “implemented the original XGBoost stage,” or “achieved independently verified 99.99% test accuracy.” Historical regression scores are paper-reported; classification “accuracy” is the wrong metric. No metric is needed in the CV bullet.

## Research CV: exact safe wording

- Co-author of a published study on hybrid learning for bifacial solar thermal collectors: Fakir et al., *Materials Research Proceedings* 64, 171–178 (2026), DOI: 10.21741/9781644904091-21.
- Implemented the deep-learning component of a πVAE–XGBoost framework proposed by Mehdi Nejjar, adapting established πVAE software to a PyTorch/Stan modelling workflow.
- Translated proposed modelling approaches into experiments, critically evaluated deep-learning choices and contributed to iterative revisions when limitations appeared.
- Contributed to the corresponding manuscript section and to experimental results and figures.

These bullets describe the historical role as personally reported. They do not claim independent reproduction of the paper's scores, authorship of the missing original tree stage, or sole development of the overall framework. Do not fold the separately AI-assisted October audit/reconstruction into this research role.

## Motivation/SOP: exact wording

In a collaborative study of bifacial solar thermal collectors, I primarily implemented and critically evaluated the deep-learning component of a πVAE–XGBoost framework proposed by Mehdi Nejjar. I adapted established πVAE software, translated modelling ideas into experiments, contributed to revisions when limitations appeared, and helped prepare the corresponding manuscript section, results and figures. Revisiting the project has reinforced the importance of separating published claims from reproducible evidence and comparing complex models with simpler alternatives. I want to develop this approach through graduate training in machine learning and scientific evaluation.

Use the role account in first person only if it remains accurate. The last sentence is proposed application framing, not a verified statement of a prior decision. Do not conflate the recent AI-assisted reconstruction with the original research work.

## Recommendation briefing for Dr. Ahmed Khallaayoun

The following are the specific parts of Mohammad's reported contribution Dr. Khallaayoun can discuss **where his supervision or records corroborate them**:

- Implementation and adaptation of the deep-learning component using established πVAE code in a PyTorch/Stan workflow.
- Translation of proposed modelling approaches into code and experiments.
- Critical evaluation of deep-learning choices and contributions to iterative model revisions; a concrete observed example would strengthen this claim.
- Contribution to the manuscript section corresponding to the deep-learning work.
- Preparation of or contribution to associated experimental results and figures; identify the actual items he observed.
- Co-authorship of the published article, with its complete author list preserved.

The source establishes the technical workflow but does not independently allocate every file or figure. Preserve the account that Mehdi suggested the overall framework and contributed to code. Original XGBoost responsibility, individual figure authorship and undocumented architecture decisions need separate corroboration. The October audit, reconstructed baselines and AI-assisted maintenance are separate work.

This is a briefing, not a drafted letter purporting to state what Dr. Khallaayoun witnessed. No message was sent to him.

## Research interview: 60–90 second answer

Our study concerned bifacial solar thermal collectors: how to estimate thermal power and outlet temperature from measured operating and environmental conditions. Mehdi Nejjar suggested the πVAE–XGBoost framework and contributed to the code. My main role was implementing and critically evaluating its deep-learning component, adapting established πVAE software, translating ideas into experiments, and contributing to revisions, writing, results and figures.

The intended pipeline predicts thermal power with πVAE, then supplies that prediction and the original inputs to XGBoost for outlet temperature. Stan samples a latent posterior through a fixed learned decoder, so its uncertainty is conditional on the trained neural weights.

The paper reported strong regression scores, but the retained historical files do not reproduce the complete final experiment. A separate, AI-assisted October reconstruction had slightly lower outlet error than direct XGBoost on one held-out day; a simple physical diagnostic was competitive, and midday interval coverage was poor. Those later findings are not paper results. Revisiting the project taught me to preserve provenance, prevent leakage, compare simple baselines, and distinguish sampler convergence from calibrated prediction.

Approximately 175 words: about 70–90 seconds at a measured speaking pace. Use the learning statements only if they reflect your own understanding; prepare to explain the diagnostics rather than imply you personally performed the automated audit unaided.

## Interview questions and answer boundaries

1. **What did you personally contribute?** Explain the account above, identify specific decisions you remember, and say which are corroborated. A current file or Git author is insufficient proof.
2. **What came from upstream πVAE?** Encoder/decoder/VAE classes and much of the feature/coefficient machinery; adaptation/integration are the defensible contribution, not architecture invention.
3. **How does Φ differ from the VAE?** Φ maps physical inputs to basis features; the VAE encodes/decodes coefficient vectors β. The resulting function is d(z)ᵀΦ(x), not a conventional VAE directly reconstructing sensor rows.
4. **Why use HMC after VAE training?** A frozen decoder converts a low-dimensional latent posterior to coefficient/function draws. Neural weights are not sampled; this is conditional uncertainty.
5. **Does this reproduce foundational πVAE prior learning?** No independent collection of prior-function draws is retained. The reconstruction uses replicas of one observed Q function and a data-dependent learned map/decoder; explain this approximation.
6. **What enters XGBoost?** In Fig. 3/methods and the reconstruction: original inputs plus scalar predicted thermal power Q, not latent z or measured Q. The historical tree code is absent.
7. **Where could leakage arise?** Historical test-conditioned Stan likelihood/checkpoint selection; combined train/test notebook scaling; in-sample upstream features in a stack; algebraic target encodings via Q/dT/outlet.
8. **Why group by day?** Strong temporal dependence makes random row splits optimistic. Whole-day Q cross-fitting does not itself produce out-of-fold outlet scores; that requires nested full-pipeline validation.
9. **What is the difference between energy and power?** Dataset Q is kW; Eq. 1 integrates energy. Do not claim to have resolved that discrepancy in the historical paper.
10. **Do the hybrid results prove superiority?** The checked 2026 one-day outlet RMSE gain is only .004823 °C; πVAE Q underperforms direct XGBoost Q and an exploratory physical-slope baseline is competitive. No general superiority claim follows.
11. **Are the 95% bands calibrated?** Credible versus observation predictive intervals differ. One-day average coverage hides poor midday coverage; temporal, neural-weight and downstream uncertainty are incomplete.
12. **Which alternatives were rejected and why?** Give remembered/corroborated historical examples, not the reconstruction's new baselines. Retained source alone cannot answer this.
13. **Can you reproduce the published 99%+ numbers?** No: original final driver/checkpoints/predictions/tree/simulator/settings are missing and historical source has a different target and test contamination. Recent saved scores can be recalculated; they are different experiments.
14. **Why not publish the measurements?** InES research consent is not a demonstrated redistribution grant. Project adaptations also need rights-holder authorization; upstream MIT is separately preserved.

## GitHub README contribution statement

Based on Mohammad Hachim Eraissouni's account, Mehdi Nejjar (@mehdinejjar86 / Night^^Stalker) suggested the overall πVAE–XGBoost framework and contributed to project code. Mohammad contributed primarily to implementation and critical evaluation of the deep-learning component, adapting established πVAE code, participating in iterative modelling revisions, contributing to the corresponding manuscript section, and preparing or contributing to experimental results and figures. He did not originate the overall framework. The retained files do not independently establish individual code or figure authorship. The publication remains co-authored by Imane Fakir, Mehdi Nejjar, Mohammad Hachim Eraissouni, Adam Kessab and Ahmed Khallaayoun. The October 2026 reconstruction and repository maintenance are separate, AI-assisted work.

Useful corroboration to obtain: Dr. Khallaayoun's observed examples, original final-Q/tree drivers and logs, manuscript revision records, a figure-creation record, and the team's permission/provenance statement. Their absence should not be filled with invented detail.
