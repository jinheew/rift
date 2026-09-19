"""the shared transition VAE"""
from __future__ import annotations

import dataclasses
import math
import statistics

import numpy as np
import torch
from torch import nn

from rift.method.transport import HIGH, LOW, reward_class_labels

VARIANTS = ("base", "bal", "bal+1p")
PRIOR_WEIGHT = 0.5          # weight of the class-prior KL term (variant bal+1p)
RECON_BALANCE = 0.5         # share of each reconstruction batch drawn from the target (bal)

GROUPS = (("A", "all"), ("B", "all"), ("A", "low"), ("A", "high"), ("B", "low"), ("B", "high"))


@dataclasses.dataclass
class VAEConfig:
    latent_dim: int = 16
    hidden_dim: int = 256
    n_hidden: int = 2
    beta: float = 0.005            # KL weight
    lr: float = 1e-3
    batch_size: int = 256
    n_steps: int = 100_000
    weight_decay: float = 0.0
    # "bal+1p" is the method in the paper
    variant: str = "bal+1p"
    reg_batch_size: int = 128      
    reg_every: int = 2          
    holdout_target_fraction: float = 0.1
    holdout_source: int = 5000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError("vae.variant must be one of {}, got {!r}".format(VARIANTS, self.variant))

    @property
    def recon_balance(self) -> float:
        return RECON_BALANCE if self.variant in ("bal", "bal+1p") else 0.0

    @property
    def prior_weight(self) -> float:
        return PRIOR_WEIGHT if self.variant == "bal+1p" else 0.0


def _mlp(in_dim: int, hidden: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TransitionVAE(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: VAEConfig):
        super().__init__()
        torch.manual_seed(config.seed)      # weight initialisation is part of the seed
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.input_dim = 2 * obs_dim + action_dim
        self.encoder = _mlp(self.input_dim, config.hidden_dim, config.n_hidden, 2 * config.latent_dim)
        self.decoder = _mlp(config.latent_dim, config.hidden_dim, config.n_hidden, self.input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x).chunk(2, dim=-1)
        return mu, log_var.clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        return self.decode(z), mu, log_var

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        recon, mu, log_var = self(x)
        recon_loss = ((recon - x) ** 2).sum(-1).mean()
        kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)).mean()
        total = recon_loss + self.config.beta * kl
        return total, {"recon": recon_loss.detach().item(), "kl": kl.detach().item(),
                       "loss": total.detach().item()}


def class_prior(quantile: float) -> tuple[float, float]:
    normal = statistics.NormalDist()
    z_q = normal.inv_cdf(1.0 - quantile)
    centre = normal.pdf(z_q) / quantile
    variance = 1.0 + z_q * centre - centre ** 2
    return float(centre), float(math.sqrt(max(variance, 1e-6)))


def train_vae(model: TransitionVAE, x: np.ndarray, context: np.ndarray, rewards: np.ndarray,
              quantile: float, config: VAEConfig, device: str = "cpu",
              log_every: int = 5_000, logger=print) -> TransitionVAE:
    torch.manual_seed(config.seed)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    data = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    pools: dict[tuple[str, str], torch.Tensor] = {}
    if config.recon_balance > 0 or config.prior_weight > 0:
        context = np.asarray(context).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        for name, value in (("A", 0), ("B", 1)):
            rows = np.flatnonzero(context == value)
            if len(rows) < 4:
                raise ValueError("context {} has {} rows; both contexts are needed".format(name, len(rows)))
            labels = reward_class_labels(rewards[rows], quantile)
            pools[(name, "all")] = torch.as_tensor(rows)
            pools[(name, "low")] = torch.as_tensor(rows[labels == LOW])
            pools[(name, "high")] = torch.as_tensor(rows[labels == HIGH])
    sizes = [config.reg_batch_size, config.reg_batch_size] + [max(2, config.reg_batch_size // 2)] * 4
    every = max(1, config.reg_every)
    prior_weight = config.prior_weight * every
    centre_q, std_q = class_prior(quantile)
    s2, log_s2 = std_q ** 2, 2.0 * math.log(std_q)

    batch = min(config.batch_size, len(data))
    n_b = int(round(batch * config.recon_balance)) if config.recon_balance > 0 else 0
    last: dict[str, float] = {}
    for step in range(1, config.n_steps + 1):
        if n_b > 0:
            pool_a, pool_b = pools[("A", "all")], pools[("B", "all")]
            idx = torch.cat([pool_a[torch.randint(0, len(pool_a), (batch - n_b,), generator=generator)],
                             pool_b[torch.randint(0, len(pool_b), (n_b,), generator=generator)]])
        else:
            idx = torch.randint(0, len(data), (batch,), generator=generator)
        loss, stats = model.loss(data[idx.to(device)])
        if prior_weight > 0 and (step % every == 0 or step == 1):
            picks = [(pools[key][torch.randint(0, len(pools[key]), (n,), generator=generator)]
                      if len(pools[key]) > 0 else pools[key][:0]) for key, n in zip(GROUPS, sizes)]
            mu, log_var = model.encode(data[torch.cat(picks).to(device)])
            offset, values, gaps = 0, [], {}
            means0: dict[tuple[str, str], torch.Tensor] = {}
            for key, pick in zip(GROUPS, picks):
                mu0, lv0 = mu[offset:offset + len(pick), 0], log_var[offset:offset + len(pick), 0]
                offset += len(pick)
                means0[key] = mu0
                if key[1] == "all" or len(mu0) == 0:
                    continue
                centre = (-1.0 if key[1] == "low" else 1.0) * centre_q
                # KL( N(mu, sigma^2) || N(centre, s_q^2) ) on coordinate 0
                values.append((0.5 * (log_s2 - lv0 + (lv0.exp() + (mu0 - centre) ** 2) / s2 - 1.0)).mean())
            for name in ("A", "B"):
                if len(means0[(name, "high")]) and len(means0[(name, "low")]):
                    gaps["prior_gap_" + name] = float((means0[(name, "high")].mean()
                                                       - means0[(name, "low")].mean()).detach())
            term = torch.stack(values).mean()
            loss = loss + prior_weight * term
            last = {"prior": float(term.detach()), **gaps}
        stats.update(last)
        stats["loss"] = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log_every and (step % log_every == 0 or step == 1 or step == config.n_steps):
            logger("  vae step {:>7d}/{}  ".format(step, config.n_steps)
                   + "  ".join("{} {:.4f}".format(k, v) for k, v in stats.items()))
    return model.eval()


@torch.no_grad()
def encode_all(model: TransitionVAE, x: np.ndarray, device: str = "cpu",
               batch_size: int = 8192) -> np.ndarray:
    model = model.to(device).eval()
    out = []
    for start in range(0, len(x), batch_size):
        chunk = torch.as_tensor(np.asarray(x[start:start + batch_size], dtype=np.float32), device=device)
        out.append(model.encode(chunk)[0].cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def decode_all(model: TransitionVAE, z: np.ndarray, device: str = "cpu",
               batch_size: int = 8192) -> np.ndarray:
    model = model.to(device).eval()
    out = []
    for start in range(0, len(z), batch_size):
        chunk = torch.as_tensor(np.asarray(z[start:start + batch_size], dtype=np.float32), device=device)
        out.append(model.decode(chunk).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
