"""the reward-improvement field, estimated in the source context"""
from __future__ import annotations

import dataclasses

import numpy as np

LOW, HIGH, MIDDLE = 0, 1, -1


@dataclasses.dataclass
class TransportConfig:
    quantile: float = 0.2         
    max_pairs: int = 2000       
    n_batches: int = 8             
    emd_max_iter: int = 1_000_000 
    ridge: float = 1e-6           
    seed: int = 0


def reward_class_labels(rewards: np.ndarray, quantile: float) -> np.ndarray:
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must lie in (0, 0.5], got {}".format(quantile))
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    low_cut = np.quantile(rewards, quantile)
    high_cut = np.quantile(rewards, 1.0 - quantile)
    labels = np.full(len(rewards), MIDDLE, dtype=np.int64)
    labels[rewards <= low_cut] = LOW
    labels[rewards >= high_cut] = HIGH
    return labels


def join(latents: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    z = np.asarray(latents, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    return np.concatenate([z, r[:, None]], axis=1)


def split(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    return u[:, :-1], u[:, -1]


@dataclasses.dataclass
class Standardizer:
    mean: np.ndarray
    whiten: np.ndarray      
    unwhiten: np.ndarray   

    @classmethod
    def fit(cls, u: np.ndarray, floor: float = 1e-6) -> "Standardizer":
        u = np.asarray(u, dtype=np.float64)
        std = np.clip(u.std(0), floor ** 0.5, None)
        return cls(u.mean(0), np.diag(1.0 / std), np.diag(std))

    def to_relative(self, u: np.ndarray) -> np.ndarray:
        return (np.asarray(u, dtype=np.float64) - self.mean) @ self.whiten.T

    def from_relative(self, u_rel: np.ndarray) -> np.ndarray:
        return np.asarray(u_rel, dtype=np.float64) @ self.unwhiten.T + self.mean


def ot_pairs(source: np.ndarray, target: np.ndarray, config: TransportConfig
             ) -> tuple[np.ndarray, np.ndarray]:
    import ot as pot

    rng = np.random.default_rng(config.seed)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        raise ValueError("optimal transport needs a non-empty low and high class")

    source_idx, target_idx = [], []
    for _ in range(max(1, config.n_batches)):
        n = min(config.max_pairs, len(source))
        m = min(config.max_pairs, len(target))
        a_idx = rng.choice(len(source), size=n, replace=len(source) < n)
        b_idx = rng.choice(len(target), size=m, replace=len(target) < m)
        cost = pot.dist(source[a_idx], target[b_idx], metric="sqeuclidean")
        cost = cost / max(cost.max(), 1e-12)        # scale-free; does not change the optimum
        wa = np.full(n, 1.0 / n)
        wb = np.full(m, 1.0 / m)
        plan, log = pot.emd(wa, wb, cost, numItermax=config.emd_max_iter, log=True)
        if log.get("warning") is not None:
            raise RuntimeError("exact OT did not converge ({}); raise emd_max_iter or lower "
                               "max_pairs".format(log["warning"]))
        best = plan.argmax(axis=1)
        mass = plan[np.arange(n), best]
        keep = mass > 0
        source_idx.append(a_idx[keep])
        target_idx.append(b_idx[best[keep]])
    return np.concatenate(source_idx), np.concatenate(target_idx)


@dataclasses.dataclass
class AffineField:
    weight: np.ndarray   
    bias: np.ndarray      

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=np.float64) @ self.weight.T + self.bias


def fit_affine_field(u: np.ndarray, v: np.ndarray, ridge: float) -> AffineField:
    u = np.asarray(u, dtype=np.float64)
    delta = np.asarray(v, dtype=np.float64) - u
    n, d = u.shape
    design = np.concatenate([u, np.ones((n, 1))], axis=1)
    gram = design.T @ design + ridge * np.eye(d + 1)
    coefficients = np.linalg.solve(gram, design.T @ delta)      # (d + 1, d)
    return AffineField(weight=coefficients[:d].T, bias=coefficients[d])


def fit_field(z_source: np.ndarray, source_rewards: np.ndarray, frame: Standardizer,
              config: TransportConfig, logger=print) -> AffineField:
    labels = reward_class_labels(source_rewards, config.quantile)
    low, high = labels == LOW, labels == HIGH
    logger("[2/3] fitting the improvement field on context A: {} low / {} high / {} middle".format(
        int(low.sum()), int(high.sum()), int((labels == MIDDLE).sum())))
    d = z_source.shape[1]
    u_low = frame.to_relative(join(z_source[low], source_rewards[low]))
    u_high = frame.to_relative(join(z_source[high], source_rewards[high]))
    i, j = ot_pairs(u_low[:, :d], u_high[:, :d], config)
    return fit_affine_field(u_low[i], u_high[j], config.ridge)
