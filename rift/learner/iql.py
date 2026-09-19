"""Implicit Q-Learning (Kostrikov et al., 2022), the downstream learner.
The implementation follows the ODRL benchmark code.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import time

import numpy as np
import torch
from torch import nn

from rift.data.datasets import Transitions


@dataclasses.dataclass
class IQLConfig:
    hidden_dim: int = 256
    n_hidden: int = 2
    discount: float = 0.99
    expectile: float = 0.7      
    beta: float = 3.0          
    max_weight: float = 100.0   
    target_update: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    n_steps: int = 1_000_000
    target_batch_fraction: float = 0.5
    cosine_actor_schedule: bool = True
    seed: int = 0


def _mlp(in_dim: int, hidden: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: IQLConfig):
        super().__init__()
        self.q1 = _mlp(obs_dim + action_dim, config.hidden_dim, config.n_hidden, 1)
        self.q2 = _mlp(obs_dim + action_dim, config.hidden_dim, config.n_hidden, 1)

    def both(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def forward(self, obs, action):
        return torch.min(*self.both(obs, action))


class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, config: IQLConfig):
        super().__init__()
        self.v = _mlp(obs_dim, config.hidden_dim, config.n_hidden, 1)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class TanhMeanPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: IQLConfig, max_action: float = 1.0):
        super().__init__()
        self.net = _mlp(obs_dim, config.hidden_dim, config.n_hidden, 2 * action_dim)
        self.max_action = max_action

    def mean(self, obs) -> torch.Tensor:
        mu, _ = self.net(obs).chunk(2, dim=-1)
        return torch.tanh(mu) * self.max_action

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.mean(obs)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


class IQL:
    def __init__(self, obs_dim: int, action_dim: int, config: IQLConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        torch.manual_seed(config.seed)
        self.critic = TwinQ(obs_dim, action_dim, config).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = ValueFunction(obs_dim, config).to(device)
        self.actor = TanhMeanPolicy(obs_dim, action_dim, config).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=config.critic_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.actor_sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, config.n_steps)
                            if config.cosine_actor_schedule else None)
        self.timing: dict[str, float] = {"train_seconds": 0.0, "eval_seconds": 0.0, "segments": 1}

    # -- checkpointing: a rerun of the same command continues where it stopped ----

    def state(self) -> dict:
        modules = {k: v.state_dict() for k, v in vars(self).items() if isinstance(v, nn.Module)}
        optimizers = {k: v.state_dict() for k, v in vars(self).items()
                      if isinstance(v, (torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler))}
        return {"modules": modules, "optimizers": optimizers}

    def load_state(self, state: dict) -> None:
        for name, payload in state["modules"].items():
            getattr(self, name).load_state_dict(payload)
        for name, payload in state["optimizers"].items():
            getattr(self, name).load_state_dict(payload)

    def _resume(self, path: str | None, generator: torch.Generator, logger,
                tag: str | None) -> tuple[int, list[dict]]:
        if not path or not os.path.exists(path):
            return 1, []
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if tag is not None and ckpt.get("tag") != tag:
            logger("  ignoring {}: written under a different configuration".format(path))
            return 1, []
        self.load_state(ckpt["agent"])
        generator.set_state(ckpt["generator"])
        rng = ckpt["torch_rng"]
        if not isinstance(rng, torch.ByteTensor):
            rng = torch.as_tensor(rng).detach().cpu().to(torch.uint8)
        torch.set_rng_state(rng)
        self.timing = {**self.timing, **ckpt.get("timing", {})}
        self.timing["segments"] = int(self.timing.get("segments", 1)) + 1
        logger("  resumed from {} at step {}".format(path, ckpt["step"]))
        return ckpt["step"] + 1, ckpt["history"]

    def _checkpoint(self, path: str, step: int, generator: torch.Generator,
                    history: list[dict], tag: str | None) -> None:
        payload = {"step": step, "agent": self.state(), "generator": generator.get_state(),
                   "torch_rng": torch.get_rng_state(), "history": history, "tag": tag,
                   "timing": dict(self.timing)}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _clock(self, base: dict, segment_started: float, eval_seconds: float) -> None:
        elapsed = time.time() - segment_started
        self.timing = {**base, "train_seconds": base["train_seconds"] + elapsed - eval_seconds,
                       "eval_seconds": base["eval_seconds"] + eval_seconds}

    # -- batches -------------------------------------------------------------

    def _batch_split(self, n: int, batch: int, target_index):
        fraction = self.config.target_batch_fraction
        if not fraction or target_index is None:
            return None
        target = torch.as_tensor(np.asarray(target_index), dtype=torch.long)
        mask = torch.ones(n, dtype=torch.bool)
        mask[target] = False
        source = mask.nonzero(as_tuple=True)[0]
        if len(target) == 0 or len(source) == 0:
            return None
        n_target = min(batch - 1, max(1, int(round(batch * fraction))))
        return target, source, n_target, batch - n_target

    @staticmethod
    def _sample(n: int, batch: int, split, generator: torch.Generator) -> torch.Tensor:
        if split is None:
            return torch.randint(0, n, (batch,), generator=generator)
        target, source, n_target, n_source = split
        return torch.cat([target[torch.randint(0, len(target), (n_target,), generator=generator)],
                          source[torch.randint(0, len(source), (n_source,), generator=generator)]])

    # -- one gradient step ------------------------------------------------------

    def _update(self, batch) -> dict[str, float]:
        obs, action, reward, next_obs, done = batch
        with torch.no_grad():
            target_q = self.critic_target(obs, action)
        v = self.value(obs)
        value_loss = expectile_loss(target_q - v, self.config.expectile).mean()
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            backup = reward + self.config.discount * (1.0 - done) * self.value(next_obs)
        q1, q2 = self.critic.both(obs, action)
        critic_loss = ((q1 - backup) ** 2 + (q2 - backup) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        with torch.no_grad():
            awr_weight = torch.exp(self.config.beta * (target_q - v)).clamp(max=self.config.max_weight)
        squared = (self.actor.mean(obs) - action) ** 2
        actor_loss = (awr_weight * squared.mean(-1)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.actor_sched is not None:
            self.actor_sched.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1.0 - self.config.target_update).add_(self.config.target_update * p)
        return {"value_loss": value_loss.item(), "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(), "q": target_q.mean().item()}

    # -- training -----------------------------------------------------------------

    def fit(self, data: Transitions, eval_fn=None, eval_every: int = 10_000, log_every: int = 10_000,
            logger=print, target_index: np.ndarray | None = None,
            checkpoint_path: str | None = None, checkpoint_every: int = 0,
            checkpoint_tag: str | None = None) -> list[dict]:
        config = self.config
        device = self.device
        tensors = (torch.as_tensor(data.observations, dtype=torch.float32, device=device),
                   torch.as_tensor(data.actions, dtype=torch.float32, device=device),
                   torch.as_tensor(data.rewards, dtype=torch.float32, device=device),
                   torch.as_tensor(data.next_observations, dtype=torch.float32, device=device),
                   torch.as_tensor(data.terminals.astype(np.float32), device=device))
        n = len(data)
        batch = min(config.batch_size, n)
        split = self._batch_split(n, batch, target_index)
        if split is not None:
            logger("  balanced batches: {} of {} rows from the {} target-side transitions".format(
                split[2], batch, len(split[0])))
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        start, history = self._resume(checkpoint_path, generator, logger, checkpoint_tag)
        base, segment_started, eval_seconds = dict(self.timing), time.time(), 0.0

        for step in range(start, config.n_steps + 1):
            idx = self._sample(n, batch, split, generator).to(device)
            stats = self._update(tuple(t[idx] for t in tensors))
            if log_every and step % log_every == 0:
                logger("  iql step {:>8d}/{}  q {:.2f}  actor {:.3f}  critic {:.3f}".format(
                    step, config.n_steps, stats["q"], stats["actor_loss"], stats["critic_loss"]))
            if eval_fn is not None and eval_every and step % eval_every == 0:
                eval_started = time.time()
                record = {"step": step, **stats, **eval_fn(self)}
                eval_seconds += time.time() - eval_started
                history.append(record)
                logger("  eval @ {:>8d}: {}".format(step, {
                    k: round(v, 2) for k, v in record.items() if k not in stats and k != "step"}))
            if checkpoint_path and checkpoint_every and step % checkpoint_every == 0:
                self._clock(base, segment_started, eval_seconds)
                self._checkpoint(checkpoint_path, step, generator, history, checkpoint_tag)
        self._clock(base, segment_started, eval_seconds)
        return history

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.actor.act(tensor).cpu().numpy().reshape(-1)
