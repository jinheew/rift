"""Load D4RL source and ODRL target datasets"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterator

import h5py
import numpy as np


@dataclasses.dataclass
class Transitions:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    episode_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.next_observations = np.asarray(self.next_observations, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        self.terminals = np.asarray(self.terminals, dtype=bool).reshape(-1)
        if self.episode_ids is not None:
            self.episode_ids = np.asarray(self.episode_ids, dtype=np.int64).reshape(-1)
        n = len(self.observations)
        for name in ("actions", "rewards", "next_observations", "terminals"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(
                    "{} has {} rows but observations has {}".format(name, got, n)
                )

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def obs_dim(self) -> int:
        return self.observations.shape[1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def select(self, index: np.ndarray) -> "Transitions":
        index = np.asarray(index)
        return Transitions(
            observations=self.observations[index],
            actions=self.actions[index],
            rewards=self.rewards[index],
            next_observations=self.next_observations[index],
            terminals=self.terminals[index],
            episode_ids=None if self.episode_ids is None else self.episode_ids[index],
        )

    def subsample(self, n: int, seed: int = 0) -> "Transitions":
        if n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        return self.select(rng.choice(len(self), size=n, replace=False))

    def episodes(self) -> Iterator[np.ndarray]:
        if self.episode_ids is None:
            yield np.arange(len(self))
            return
        order = np.argsort(self.episode_ids, kind="stable")
        ids = self.episode_ids[order]
        boundaries = np.flatnonzero(np.diff(ids)) + 1
        for chunk in np.split(order, boundaries):
            if len(chunk):
                yield chunk


def concatenate(parts: list[Transitions]) -> Transitions:
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("nothing to concatenate")
    ids, offset = [], 0
    for p in parts:
        if p.episode_ids is None:
            ids.append(np.full(len(p), -1, dtype=np.int64))
        else:
            ids.append(p.episode_ids + offset)
            offset += int(p.episode_ids.max()) + 1
    return Transitions(
        observations=np.concatenate([p.observations for p in parts]),
        actions=np.concatenate([p.actions for p in parts]),
        rewards=np.concatenate([p.rewards for p in parts]),
        next_observations=np.concatenate([p.next_observations for p in parts]),
        terminals=np.concatenate([p.terminals for p in parts]),
        episode_ids=np.concatenate(ids),
    )


_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
    "next_observations",
)


def _read(path: pathlib.Path | str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as fh:
        for name in _FIELDS:
            node = fh.get(name)
            if isinstance(node, h5py.Dataset) and node.ndim > 0:
                out[name] = node[:]
    missing = {"observations", "actions", "rewards", "terminals"} - set(out)
    if missing:
        raise ValueError("{} is missing required fields {}".format(path, sorted(missing)))
    return out


# ignore all zero state cols, not used in D4RL
ANT_STATE_DIM = 27

def _maybe_truncate_ant(task: str, obs: np.ndarray) -> np.ndarray:
    if task != "ant" or obs.shape[1] <= ANT_STATE_DIM:
        return obs
    tail = obs[:, ANT_STATE_DIM:]
    if np.abs(tail).max() > 0:
        raise ValueError(
            "ant contact-force columns are not all zero (max |x| = {:.3e}); "
            "truncating to {} dims would lose information".format(
                np.abs(tail).max(), ANT_STATE_DIM)
        )
    return obs[:, :ANT_STATE_DIM]


def load_source(path: pathlib.Path | str, task: str) -> Transitions:
    raw = _read(path)
    obs = _maybe_truncate_ant(task, np.asarray(raw["observations"]))
    actions = np.asarray(raw["actions"])
    rewards = np.asarray(raw["rewards"]).reshape(-1)
    terminals = np.asarray(raw["terminals"]).reshape(-1).astype(bool)
    timeouts = (np.asarray(raw["timeouts"]).reshape(-1).astype(bool)
                if "timeouts" in raw else np.zeros(len(obs), dtype=bool))

    if "next_observations" in raw:
        next_obs = _maybe_truncate_ant(task, np.asarray(raw["next_observations"]))
        keep = np.ones(len(obs), dtype=bool)
    else:
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
        keep = ~(terminals | timeouts)
        keep[-1] = False

    ends = terminals | timeouts
    episode_ids = np.concatenate([[0], np.cumsum(ends[:-1])]).astype(np.int64)

    data = Transitions(obs, actions, rewards, next_obs, terminals, episode_ids)
    return data.select(np.flatnonzero(keep))


def implied_timeouts(obs: np.ndarray, next_obs: np.ndarray,
                     tol: float = 1e-6) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    flags = np.zeros(len(obs), dtype=bool)
    if len(obs) > 1:
        flags[:-1] = np.abs(next_obs[:-1] - obs[1:]).max(axis=1) > tol
    return flags


def load_target(path: pathlib.Path | str, task: str) -> Transitions:
    raw = _read(path)
    obs = _maybe_truncate_ant(task, np.asarray(raw["observations"]))
    terminals = np.asarray(raw["terminals"]).reshape(-1).astype(bool)
    if "next_observations" in raw:
        next_obs = _maybe_truncate_ant(task, np.asarray(raw["next_observations"]))
        keep = np.ones(len(obs), dtype=bool)
    else: 
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
        keep = ~terminals
        keep[-1] = False

    if "timeouts" in raw:
        timeouts = np.asarray(raw["timeouts"]).reshape(-1).astype(bool)
    else:
        timeouts = implied_timeouts(obs, next_obs)
    ends = terminals | timeouts
    episode_ids = np.concatenate([[0], np.cumsum(ends[:-1])]).astype(np.int64)

    data = Transitions(
        obs,
        np.asarray(raw["actions"]),
        np.asarray(raw["rewards"]).reshape(-1),
        next_obs,
        terminals,
        episode_ids,
    )
    return data.select(np.flatnonzero(keep))


def make_medium_expert(medium: Transitions, expert: Transitions,
                       n_medium: int = 2, n_expert: int = 3,
                       seed: int = 0) -> Transitions:
    rng = np.random.default_rng(seed)
    picked = []
    for data, count in ((medium, n_medium), (expert, n_expert)):
        episodes = list(data.episodes())
        if not episodes:
            continue
        take = rng.choice(len(episodes), size=min(count, len(episodes)), replace=False)
        picked.extend(data.select(episodes[int(i)]) for i in take)
    return concatenate(picked)


@dataclasses.dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, *arrays: np.ndarray, eps: float = 1e-3) -> "Normalizer":
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        return cls(stacked.mean(0).astype(np.float32),
                   (stacked.std(0) + eps).astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)
