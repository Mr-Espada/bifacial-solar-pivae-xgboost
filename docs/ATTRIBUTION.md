# Attribution

The research publication is:

Imane Fakir, Mehdi Nejjar, Mohammad Hachim Eraissouni, Adam Kessab, Ahmed Khallaayoun. “A Hybrid πVAE-XGBoost Framework for High-Fidelity Simulation of Bifacial Solar Thermal Collectors.” *Materials Research Proceedings* 64 (2026), 171–178. [DOI](https://doi.org/10.21741/9781644904091-21).

Repository credits are listed in [CONTRIBUTORS.md](../CONTRIBUTORS.md).

## πVAE

The historical neural implementation adapts [MLGlobalHealth/pi-vae](https://github.com/MLGlobalHealth/pi-vae), reviewed at commit `14de0377500ce175115a6509a802e473c3466a54`. The reconstruction follows the same source lineage.

The full upstream MIT notice is retained in [third_party/pi-vae/LICENSE](../third_party/pi-vae/LICENSE), copyright © 2022 Machine Learning and Global Health Network.

Method reference: S. Mishra, S. Flaxman, T. Berah, H. Zhu, M. Pakkanen and S. Bhatt. “πVAE: A stochastic process prior for Bayesian deep learning with MCMC.” *Statistics and Computing* 32, 96 (2022). [DOI](https://doi.org/10.1007/s11222-022-10151-w).

## XGBoost and other dependencies

T. Chen and C. Guestrin. “XGBoost: A Scalable Tree Boosting System” (2016). [Paper](https://arxiv.org/abs/1603.02754).

PyTorch, Stan/CmdStanPy, XGBoost and the numerical Python packages are installed separately and retain their own licenses.

The publication attributes the measurements to the Institute of New Energy Systems (InES), Germany. Measurement files and trained artifacts are excluded; see [DATA.md](DATA.md). Project licensing is described in [LICENSE.md](../LICENSE.md).
