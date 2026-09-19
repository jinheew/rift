"""run one cell: augment the target data with RIFT, train IQL, evaluate in the target"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any

import numpy as np

from rift.cache import ArtifactCache
from rift.data import sources
from rift.data.datasets import Transitions, concatenate, load_source, load_target, make_medium_expert
from rift.data.sources import fetch_source, fetch_target
from rift.learner.evaluate import evaluate_policy
from rift.learner.iql import IQL, IQLConfig
from rift.method.augment import build_augmentation
from rift.method.transport import TransportConfig
from rift.method.vae import VAEConfig


@dataclasses.dataclass
class RunConfig:
    task: str = "hopper"
    shift: str = "gravity"
    level: float | str = 0.5
    source_quality: str = "medium"
    target_quality: str = "medium"
    seed: int = 0
    device: str = "cpu"
    data_seed: int | None = 0
    n_target: int | None = 5000
    # The training-set arm. False: IQL on D_B u D_B'. True: IQL on D_A u D_B u D_B'
    include_source: bool = False

    n_eval_episodes: int = 10
    eval_every: int = 10_000
    log_every: int = 10_000
    checkpoint_every: int = 50_000     
    cache_dir: str | None = "results/cache" 

    iql: dict[str, Any] = dataclasses.field(default_factory=dict)
    vae: dict[str, Any] = dataclasses.field(default_factory=dict)
    transport: dict[str, Any] = dataclasses.field(default_factory=dict)
    output_dir: str = "results"

    @property
    def arm(self) -> str:
        return "AB" if self.include_source else "B"

    @property
    def cell_name(self) -> str:
        return "{}_{}_{}_{}_{}_nT{}_seed{}".format(self.task, self.shift, self.level, self.source_quality,
                                                  self.target_quality, self.n_target or "all", self.seed)

    @property
    def run_name(self) -> str:
        return "{}_{}_{}_{}_{}_rift-{}_seed{}".format(self.task, self.shift, self.level, self.source_quality,
                                                     self.target_quality, self.arm, self.seed)


def load_datasets(config: RunConfig, logger=print) -> tuple[Transitions, Transitions]:
    data_seed = config.seed if config.data_seed is None else config.data_seed
    source = load_source(fetch_source(config.task, config.source_quality), config.task)
    if config.target_quality == "medium_expert":
        # ODRL ships random / medium / expert; the protocol builds medium-expert
        # from 2 medium and 3 expert trajectories.
        medium = load_target(fetch_target(config.task, config.shift, config.level, "medium"), config.task)
        expert = load_target(fetch_target(config.task, config.shift, config.level, "expert"), config.task)
        target = make_medium_expert(medium, expert, seed=data_seed)
    else:
        target = load_target(fetch_target(config.task, config.shift, config.level, config.target_quality),
                             config.task)
    if config.n_target:
        target = target.subsample(config.n_target, seed=data_seed)
    logger("source {} ({}) n={}   target {} {} {} ({}) n={}".format(
        config.task, config.source_quality, len(source), config.task, config.shift, config.level,
        config.target_quality, len(target)))
    return source, target


def run(config: RunConfig, logger=print) -> dict:
    started = time.time()
    np.random.seed(config.seed)
    source, target = load_datasets(config, logger=logger)
    loaded = time.time()

    iql_config = IQLConfig(seed=config.seed, **config.iql)
    vae_config = VAEConfig(seed=config.seed, **config.vae)
    transport_config = TransportConfig(seed=config.seed, **config.transport)
    cache = ArtifactCache(pathlib.Path(config.cache_dir), config.cell_name) if config.cache_dir else None
    out_dir = pathlib.Path(config.output_dir)
    checkpoint = str(out_dir / "checkpoints" / (config.run_name + ".pt")) if config.checkpoint_every else None
    # The checkpoint tag ties it to the exact configuration that wrote it,
    # every dataclass default included.
    resolved = {k: v for k, v in dataclasses.asdict(config).items() if k != "cache_dir"}
    resolved.update({"iql": dataclasses.asdict(iql_config), "vae": dataclasses.asdict(vae_config),
                     "transport": dataclasses.asdict(transport_config)})
    tag = json.dumps(resolved, sort_keys=True, default=str)

    artifacts = build_augmentation(source, target, config.task, vae_config=vae_config,
                                   transport_config=transport_config, device=config.device,
                                   logger=logger, cache=cache)
    extra: dict[str, Any] = dict(artifacts.stats)
    target_side = concatenate([target, artifacts.synthetic]) if len(artifacts.synthetic) else target
    training = concatenate([source, target_side]) if config.include_source else target_side
    training = artifacts.standardized(training)          # stage 4: the learner's coordinates
    target_index = np.arange(len(training) - len(target_side), len(training))
    preprocess_seconds = time.time() - loaded

    def eval_fn(agent) -> dict[str, float]:
        return evaluate_policy(agent, config.task, config.shift, config.level,
                               n_episodes=config.n_eval_episodes, seed=config.seed,
                               observation_transform=artifacts.observation_transform)

    logger("training IQL on {} transitions ({} target-side)".format(len(training), len(target_side)))
    agent = IQL(training.obs_dim, training.action_dim, iql_config, device=config.device)
    history = agent.fit(training, eval_fn=eval_fn, eval_every=config.eval_every, log_every=config.log_every,
                        logger=logger, target_index=target_index,
                        checkpoint_path=checkpoint, checkpoint_every=config.checkpoint_every,
                        checkpoint_tag=tag)
    extra["n_training"] = float(len(training))
    # A critic that has blown up can still give a plausible score (the AWR
    # weights are clipped); record the magnitude so such runs stay visible.
    q_values = [abs(r["q"]) for r in history if r.get("q") is not None]
    if q_values:
        extra["q_max"] = float(max(q_values))
        extra["diverged"] = float(extra["q_max"] > 1e5)

    final = eval_fn(agent)
    finished = time.time()
    result = {
        "config": dataclasses.asdict(config),
        "final": final,
        "history": history,
        "extra": extra,
        "timing": {"load_seconds": loaded - started, "preprocess_seconds": preprocess_seconds,
                   "cache_hits": dict(cache.hits) if cache is not None else {},
                   "train_seconds": float(agent.timing["train_seconds"]),
                   "eval_seconds": float(agent.timing["eval_seconds"]),
                   "elapsed_seconds": finished - started},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / (config.run_name + ".json")).write_text(json.dumps(result, indent=2))
    if checkpoint:
        pathlib.Path(checkpoint).unlink(missing_ok=True)
    logger("{}: normalized_score {:.2f}  return {:.1f}  ({:.1f} s)".format(
        config.run_name, final["normalized_score"], final["return_mean"], finished - started))
    return result
