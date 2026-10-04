"""Neural fitting uses training rows only, with fixed epochs and local logging."""

import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import PIVAE, calculate_loss


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fit_neural(x, y, features, cfg, seed, output):
    """No validation/test argument exists; preprocessing is fitted here only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    opts = cfg["pivae"]
    seed_everything(seed, opts["threads"])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    x_scale = np.where(x_scale > 0, x_scale, 1.0)
    y_mean, y_scale = float(y.mean()), float(y.std())
    if y_scale <= 0:
        raise ValueError("Q training target must vary")
    xs = torch.from_numpy(((x - x_mean) / x_scale).astype(np.float32))
    ys = torch.from_numpy(((y - y_mean) / y_scale).astype(np.float32))
    model = PIVAE(len(features), opts)
    generator = torch.Generator().manual_seed(seed)
    if opts["n_centers"] > len(x):
        raise ValueError("More RBF centers than fitting rows")
    centers = torch.randperm(len(x), generator=generator)[:opts["n_centers"]]
    with torch.no_grad():
        model.phi.centers.copy_(xs[centers])
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["learning_rate"])
    history = []
    start = time.monotonic()
    for epoch in range(opts["epochs"]):
        model.train()
        sums = np.zeros(3)
        order = torch.randperm(len(x), generator=generator)
        for batch in order.split(opts["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            direct, reconstructed, mu, logvar = model(xs[batch])
            loss, reconstruction, kl = calculate_loss(ys[batch], direct, reconstructed, mu, logvar, opts["kl_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite piVAE loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts["gradient_clip"])
            optimizer.step()
            sums += np.array([loss.item(), reconstruction.item(), kl.item()]) * len(batch)
        history.append({"epoch": epoch + 1, "loss": float(sums[0] / len(x)),
                        "reconstruction": float(sums[1] / len(x)), "kl": float(sums[2] / len(x))})
        if epoch == 0 or (epoch + 1) % opts["log_every"] == 0 or epoch + 1 == opts["epochs"]:
            print(f"{output.name}: epoch {epoch+1}/{opts['epochs']}, loss={history[-1]['loss']:.6f}, elapsed={time.monotonic()-start:.1f}s", flush=True)
    model.eval()
    checkpoint = {
        "state_dict": model.state_dict(), "pivae_config": opts, "features": list(features),
        "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(),
        "y_mean": y_mean, "y_scale": y_scale, "seed": seed,
    }
    torch.save(checkpoint, output / "neural.pt")
    pd.DataFrame(history).to_csv(output / "loss.csv", index=False)
    return model, checkpoint


def load_neural(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_num_threads(state["pivae_config"]["threads"])
    model = PIVAE(len(state["features"]), state["pivae_config"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def feature_matrix(model, checkpoint, x, batch_size=4096):
    xs = ((np.asarray(x, dtype=np.float64) - checkpoint["x_mean"]) / checkpoint["x_scale"]).astype(np.float32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(xs), batch_size):
            chunks.append(model.phi(torch.from_numpy(xs[start:start+batch_size])).numpy().astype(np.float64))
    return np.concatenate(chunks)
