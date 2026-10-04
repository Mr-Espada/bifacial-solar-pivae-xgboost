"""Training-only Stan/HMC posterior; no targets enter prediction."""

import hashlib
from pathlib import Path
import shutil
import numpy as np
import torch
from .data import write_json
from .data import file_hash
from .training import feature_matrix


def compress_gaussian_likelihood(phi, y):
    """For any beta, SSE = ||R beta - qty||^2 + residual_sse exactly.

    QR of the augmented [Phi,y] matrix is numerically safer than subtracting
    large nearly equal terms in Gram sufficient statistics. No rows are thinned.
    """
    phi = np.asarray(phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = phi.shape
    if n <= p or len(y) != n:
        raise ValueError("QR compression needs more observations than basis coefficients")
    r = np.linalg.qr(np.column_stack([phi, y]), mode="r")
    return {"n_train": n, "R": r[:p, :p], "qty": r[:p, p], "residual_sse": float(r[p, p] ** 2)}


def build_stan_model(run_dir):
    import cmdstanpy
    source = Path(__file__).with_name("pivae_2d.stan")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    directory = Path(run_dir) / "stan_build" / digest
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "pivae_2d.stan"
    if not target.exists():
        shutil.copyfile(source, target)
    try:
        return cmdstanpy.CmdStanModel(stan_file=str(target), cpp_options={"O": "1", "PRECOMPILED_HEADERS": "false"})
    except ValueError as exc:
        raise RuntimeError("CmdStan is required. Run scripts/install_cmdstan.py, then set CMDSTAN if needed.") from exc


def condition_posterior(model, checkpoint, x_train, y_train, cfg, seed, output, stan_model):
    """Only fitting rows, never the rows being predicted, form this likelihood."""
    output = Path(output)
    phi = feature_matrix(model, checkpoint, x_train)
    y = (np.asarray(y_train, float) - checkpoint["y_mean"]) / checkpoint["y_scale"]
    data = compress_gaussian_likelihood(phi, y)
    np.savez_compressed(output / "training_likelihood.npz", **data)
    decoder = model.vae.decoder
    opts = cfg["pivae"]
    data.update({"p": opts["latent_dim"], "p1": opts["vae_hidden"][0],
                 "p2": opts["vae_hidden"][1], "beta_dim": opts["beta_dim"],
                 "sigma_prior_mean": cfg["posterior"]["sigma_prior_mean"],
                 "sigma_prior_sd": cfg["posterior"]["sigma_prior_sd"]})
    for i, layer in enumerate((decoder.linear1, decoder.linear2, decoder.out), 1):
        data[f"W{i}"] = layer.weight.detach().numpy().T.astype(np.float64)
        data[f"B{i}"] = layer.bias.detach().numpy().astype(np.float64)
    with torch.inference_mode():
        mu, _ = model.vae.encoder(model.betas)
        init_z = mu.mean(dim=0).numpy().astype(float)
        init_beta = model.vae.decoder(torch.from_numpy(init_z.astype(np.float32))).numpy()
    residual = phi @ init_beta - y
    init_sigma = max(0.002, float(np.sqrt(np.mean(residual**2))))
    rng = np.random.default_rng(seed)
    hmc = cfg["posterior"]
    center = {"z": init_z.tolist(), "sigma": init_sigma}
    if hmc.get("initialization", "encoder") == "multistart_map":
        # This is an initialization aid, never a replacement for posterior draws.
        # Dispersed starts reveal local modes of this training-only likelihood.
        modes = []
        best = None
        for attempt in range(hmc["map_starts"]):
            attempt_dir = output / "initialization" / f"start_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            initial = {"z": (init_z + rng.normal(0, 1.0, len(init_z))).tolist(), "sigma": init_sigma}
            try:
                mode = stan_model.optimize(data=data, seed=seed+attempt, inits=initial,
                                           algorithm="lbfgs", iter=hmc["map_iters"],
                                           jacobian=True, show_console=False, sig_figs=16,
                                           output_dir=str(attempt_dir))
                values = mode.optimized_params_dict
                log_density = float(values["lp__"])
                z_mode = mode.stan_variable("z").tolist()
                sigma_mode = float(mode.stan_variable("sigma"))
                modes.append({"attempt": attempt, "seed": seed+attempt, "lp": log_density,
                              "z": z_mode, "sigma": sigma_mode,
                              "outputs": {p.name: file_hash(p) for p in sorted(attempt_dir.iterdir()) if p.is_file()}})
                if best is None or log_density > best["lp"]:
                    best = modes[-1]
            except RuntimeError as exc:
                modes.append({"attempt": attempt, "seed": seed+attempt, "error": str(exc)})
        if best is None:
            raise RuntimeError("Every training-only MAP initialization failed")
        center = {"z": best["z"], "sigma": best["sigma"]}
        write_json(output / "initialization.json", {"method": "training-only dispersed multi-start MAP",
                   "modes": modes, "selected_lp": best["lp"], "selected_attempt": best["attempt"],
                   "limit": "Initializes HMC only. Rhat/ESS do not establish global exploration of every mode."})
    jitter = hmc.get("init_jitter", 0.1)
    inits = [{"z": (np.asarray(center["z"]) + rng.normal(0, jitter, len(init_z))).tolist(),
              "sigma": center["sigma"]} for _ in range(hmc["chains"])]
    print(f"{output.name}: HMC on all {len(y)} fitting rows (exact QR likelihood)", flush=True)
    fit = stan_model.sample(
        data=data, seed=seed, chains=hmc["chains"], parallel_chains=hmc["parallel_chains"],
        iter_warmup=hmc["warmup"], iter_sampling=hmc["samples"], adapt_delta=hmc["adapt_delta"],
        max_treedepth=hmc["max_treedepth"], metric=hmc.get("metric", "diag_e"),
        inits=inits, show_progress=False, sig_figs=16,
        refresh=hmc["warmup"] + hmc["samples"], output_dir=str(output / "hmc"),
    )
    beta, sigma, z = (fit.stan_variable(name) for name in ("f", "sigma", "z"))
    np.savez_compressed(output / "posterior.npz", beta=beta, sigma=sigma, z=z)
    summary = fit.summary()
    summary.to_csv(output / "posterior_summary.csv")
    method = fit.method_variables()
    energies = method["energy__"]
    bfmi = np.mean(np.diff(energies, axis=0)**2, axis=0) / np.var(energies, axis=0)
    relevant = summary.loc[[i for i in summary.index if i == "sigma" or i.startswith(("z[", "f["))]]
    rhat = float(relevant["R_hat"].max())
    ess = float(relevant["ESS_bulk"].min())
    all_finite = bool(np.isfinite(relevant[["R_hat", "ESS_bulk", "ESS_tail"]].to_numpy()).all())
    if not all_finite:
        raise RuntimeError("Nonfinite posterior diagnostics; no downstream model should use this fit")
    diagnostics = {
        "fitting_rows": len(y), "draws": len(beta), "chains": hmc["chains"],
        "max_rhat": rhat, "min_bulk_ess": ess, "bfmi_per_chain": bfmi.tolist(),
        "all_relevant_diagnostics_finite": all_finite,
        "min_tail_ess": float(relevant["ESS_tail"].min()),
        "divergences": int(method["divergent__"].sum()),
        "max_treedepth_hits": int((method["treedepth__"] >= hmc["max_treedepth"]).sum()),
        "rhat_threshold": hmc["rhat_threshold"], "min_ess_threshold": hmc["min_ess"],
    }
    diagnostics["quality_passed"] = bool(
        all_finite and np.isfinite(rhat) and rhat <= hmc["rhat_threshold"] and ess >= hmc["min_ess"]
        and diagnostics["divergences"] == 0 and diagnostics["max_treedepth_hits"] == 0
        and np.all(bfmi > 0.3)
    )
    write_json(output / "diagnostics.json", diagnostics)
    (output / "diagnose.txt").write_text(fit.diagnose())
    print(f"{output.name}: Rhat={rhat:.3f}, min ESS={ess:.0f}, divergences={diagnostics['divergences']}, quality={diagnostics['quality_passed']}", flush=True)
    if not diagnostics["quality_passed"]:
        raise RuntimeError("Posterior quality gate failed; diagnostics and logs retained, downstream use refused")
    return {"beta": beta, "sigma": sigma, "z": z}


def predict_posterior(model, checkpoint, x, posterior, alpha=0.05, chunk_size=512, seed=0):
    """No target argument. Point prediction is E[d(z)] dot Phi(x)."""
    phi = feature_matrix(model, checkpoint, x)
    beta = np.asarray(posterior["beta"], float)
    sigma = np.asarray(posterior["sigma"], float)
    if beta.ndim != 2 or sigma.shape != (len(beta),) or beta.shape[1] != phi.shape[1]:
        raise ValueError("Posterior dimensions do not match the frozen model")
    mean = phi @ beta.mean(axis=0)
    ci_low, ci_high, pi_low, pi_high = (np.empty(len(phi)) for _ in range(4))
    rng = np.random.default_rng(seed)
    for start in range(0, len(phi), chunk_size):
        end = min(start + chunk_size, len(phi))
        latent = beta @ phi[start:end].T
        ci_low[start:end], ci_high[start:end] = np.quantile(latent, [alpha/2, 1-alpha/2], axis=0)
        observation = latent + rng.normal(size=latent.shape) * sigma[:, None]
        pi_low[start:end], pi_high[start:end] = np.quantile(observation, [alpha/2, 1-alpha/2], axis=0)
    scale, offset = checkpoint["y_scale"], checkpoint["y_mean"]
    return {
        "predicted_Q_1kW": mean * scale + offset,
        "q_ci_low": ci_low * scale + offset, "q_ci_high": ci_high * scale + offset,
        "q_pi_low": pi_low * scale + offset, "q_pi_high": pi_high * scale + offset,
    }
