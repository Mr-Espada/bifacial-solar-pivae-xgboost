"""Refactored RBF/tanh piVAE, preserving coefficient and decoder structure."""

import torch
from torch import nn


class PHI(nn.Module):
    def __init__(self, inputs, centers, hidden, outputs, alpha):
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(centers, inputs))
        self.alpha = alpha
        self.linear1 = nn.Linear(centers, hidden[0])
        self.linear2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], outputs)

    def forward(self, x):
        rbf = torch.exp(-self.alpha * torch.cdist(x, self.centers).square())
        return self.out(torch.tanh(self.linear2(torch.tanh(self.linear1(rbf)))))


class Encoder(nn.Module):
    def __init__(self, inputs, hidden, latent):
        super().__init__()
        self.linear1 = nn.Linear(inputs, hidden[0])
        self.linear2 = nn.Linear(hidden[0], hidden[1])
        self.mu = nn.Linear(hidden[1], latent)
        self.logvar = nn.Linear(hidden[1], latent)

    def forward(self, beta):
        hidden = torch.tanh(self.linear2(torch.tanh(self.linear1(beta))))
        return self.mu(hidden), self.logvar(hidden)


class Decoder(nn.Module):
    def __init__(self, latent, hidden, outputs):
        super().__init__()
        self.linear1 = nn.Linear(latent, hidden[0])
        self.linear2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], outputs)

    def forward(self, z):
        return self.out(torch.tanh(self.linear2(torch.tanh(self.linear1(z)))))


class VAE(nn.Module):
    def __init__(self, beta_dim, hidden, latent):
        super().__init__()
        self.encoder = Encoder(beta_dim, hidden, latent)
        self.decoder = Decoder(latent, hidden, beta_dim)

    def forward(self, beta):
        mu, logvar = self.encoder(beta)
        z = mu + torch.exp(logvar / 2) * torch.randn_like(mu) if self.training else mu
        return self.decoder(z), mu, logvar


class PIVAE(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.phi = PHI(input_dim, cfg["n_centers"], cfg["feature_hidden"], cfg["beta_dim"], cfg["rbf_alpha"])
        # Each row is a persistent coefficient replica for the same Q function.
        # It is never assigned to a shuffled row's minibatch slot.
        self.betas = nn.Parameter(torch.randn(cfg["coefficient_replicas"], cfg["beta_dim"]) * 0.05)
        self.vae = VAE(cfg["beta_dim"], cfg["vae_hidden"], cfg["latent_dim"])

    def forward(self, x):
        phi = self.phi(x)
        reconstructed, mu, logvar = self.vae(self.betas)
        return self.betas @ phi.T, reconstructed @ phi.T, mu, logvar


def calculate_loss(target, direct, reconstructed, mu, logvar, kl_weight):
    target = target.reshape(1, -1)
    reconstruction = (direct - target).square().mean() + (reconstructed - target).square().mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()
    return reconstruction + kl_weight * kl, reconstruction, kl
