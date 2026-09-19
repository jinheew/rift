"""Policy evaluation in the shifted target environment"""
from __future__ import annotations

import numpy as np

from rift.envs.reference_scores import normalized_score
from rift.envs.shifted import make_env

# ODRL wraps its locomotion environments in a 1000-step time limit.
MAX_EPISODE_STEPS = 1000


def evaluate_policy(agent, task: str, shift: str, level: float | str,
                    n_episodes: int = 10, seed: int = 0, observation_transform=None,
                    asset_dir=None, max_steps: int = MAX_EPISODE_STEPS) -> dict[str, float]:
    transform = observation_transform or (lambda x: x)
    env = make_env(task, shift=shift, level=level, asset_dir=asset_dir)
    returns, lengths = [], []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, steps = 0.0, 0
        for steps in range(1, max_steps + 1):
            obs, reward, terminated, truncated, _ = env.step(agent.act(transform(obs)))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
    env.close()

    returns_arr = np.asarray(returns, dtype=np.float64)
    result = {
        "return_mean": float(returns_arr.mean()),
        "return_std": float(returns_arr.std()),
        "episode_length": float(np.mean(lengths)),
    }
    try:
        result["normalized_score"] = float(
            normalized_score(task, shift, level, result["return_mean"])
        )
    except KeyError:
        result["normalized_score"] = float("nan")
    return result
