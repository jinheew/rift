"""RIFT end to end"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from rift.data.datasets import Normalizer, Transitions, concatenate
from rift.envs.shifted import terminal_flags
from rift.method import transport as T
from rift.method.vae import TransitionVAE, VAEConfig, decode_all, encode_all, train_vae

ACTION_LOW, ACTION_HIGH = -1.0, 1.0 


@dataclasses.dataclass
class TransitionNormalizer:
    """Standardises ``[s, a, s']`` for the VAE"""

    obs: Normalizer
    action: Normalizer

    @classmethod
    def fit(cls, data: Transitions) -> "TransitionNormalizer":
        return cls(obs=Normalizer.fit(data.observations, data.next_observations),
                   action=Normalizer.fit(data.actions))

    def pack(self, data: Transitions) -> np.ndarray:
        return np.concatenate([self.obs(data.observations), self.action(data.actions),
                               self.obs(data.next_observations)], axis=1).astype(np.float32)

    def unpack(self, packed: np.ndarray, obs_dim: int, action_dim: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        o, a = obs_dim, action_dim
        return (self.obs.inverse(packed[:, :o]), self.action.inverse(packed[:, o:o + a]),
                self.obs.inverse(packed[:, o + a:]))


@dataclasses.dataclass
class AugmentationArtifacts:
    synthetic: Transitions            
    vae: TransitionVAE
    normalizer: TransitionNormalizer  
    field: T.AffineField
    frame_source: T.Standardizer
    frame_target: T.Standardizer
    stats: dict[str, float]

    @property
    def observation_transform(self):
        return self.normalizer.obs

    def standardized(self, data: Transitions) -> Transitions:
        return Transitions(self.normalizer.obs(data.observations), data.actions, data.rewards,
                           self.normalizer.obs(data.next_observations), data.terminals, data.episode_ids)


def _stage_key(vae_config: VAEConfig, transport_config: T.TransportConfig) -> dict:
    return {"vae": dataclasses.asdict(vae_config), "transport": dataclasses.asdict(transport_config)}


def _training_rows(source: Transitions, target: Transitions, config: VAEConfig
                   ) -> tuple[Transitions, Transitions]:
    rng = np.random.default_rng(config.seed)
    perm_b = rng.permutation(len(target))
    n_hold_b = (max(1, int(config.holdout_target_fraction * len(target)))
                if config.holdout_target_fraction > 0 else 0)
    perm_a = rng.permutation(len(source))
    n_hold_a = min(config.holdout_source, len(source) // 10)
    return source.select(perm_a[n_hold_a:]), target.select(perm_b[n_hold_b:])


def build_augmentation(source: Transitions, target: Transitions, task: str,
                       vae_config: VAEConfig | None = None,
                       transport_config: T.TransportConfig | None = None,
                       device: str = "cpu", logger=print, cache=None) -> AugmentationArtifacts:
    vae_config = vae_config or VAEConfig()
    transport_config = transport_config or T.TransportConfig()
    mixed = concatenate([source, target])
    normalizer = TransitionNormalizer.fit(mixed)
    model = TransitionVAE(mixed.obs_dim, mixed.action_dim, vae_config)

    stage_key = _stage_key(vae_config, transport_config)
    cached = cache.load("rift_vae_transport", stage_key) if cache is not None else None
    if cached is not None:
        logger("[1/3] VAE and [2/3] field loaded from {}".format(cache.path("rift_vae_transport", stage_key)))
        model.load_state_dict(cached["vae_state"])
        model.to(device).eval()
        frame_source = T.Standardizer(**cached["frame_source"])
        frame_target = T.Standardizer(**cached["frame_target"])
        field = T.AffineField(**cached["field"])
        z_target = encode_all(model, normalizer.pack(target), device=device)
    else:
        started = time.time()
        vae_only = cache.load("rift_vae", dataclasses.asdict(vae_config)) if cache is not None else None
        if vae_only is not None:
            logger("[1/3] VAE loaded from {} (transport settings differ: refitting the field)".format(
                cache.path("rift_vae", dataclasses.asdict(vae_config))))
            model.load_state_dict(vae_only["vae_state"])
            model.to(device).eval()
        else:
            source_train, target_train = _training_rows(source, target, vae_config)
            train = concatenate([source_train, target_train])
            context = np.r_[np.zeros(len(source_train)), np.ones(len(target_train))]
            logger("[1/3] training the shared VAE ({}) on {} A + {} B transitions for {} steps".format(
                vae_config.variant, len(source_train), len(target_train), vae_config.n_steps))
            train_vae(model, normalizer.pack(train), context, train.rewards, transport_config.quantile,
                      vae_config, device=device, logger=logger)
            if cache is not None:
                cache.save("rift_vae", dataclasses.asdict(vae_config),
                           {"vae_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                           compute_seconds=time.time() - started)
        z_source = encode_all(model, normalizer.pack(source), device=device)
        z_target = encode_all(model, normalizer.pack(target), device=device)
        frame_source = T.Standardizer.fit(T.join(z_source, source.rewards))
        frame_target = T.Standardizer.fit(T.join(z_target, target.rewards))
        field = T.fit_field(z_source, source.rewards, frame_source, transport_config, logger=logger)
        if cache is not None:
            cache.save("rift_vae_transport", stage_key, {
                "vae_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "frame_source": dataclasses.asdict(frame_source),
                "frame_target": dataclasses.asdict(frame_target),
                "field": dataclasses.asdict(field)}, compute_seconds=time.time() - started)

    labels = T.reward_class_labels(target.rewards, transport_config.quantile)
    anchors = np.flatnonzero(labels == T.LOW)
    u_anchor = T.join(z_target[anchors], target.rewards[anchors])
    logger("[3/3] transporting {} low-reward target anchors (q = {})".format(
        len(anchors), transport_config.quantile))
    u_rel = frame_target.to_relative(u_anchor)
    moved = frame_target.from_relative(u_rel + field(u_rel))
    z_new, r_new = T.split(moved)
    decoded = decode_all(model, z_new.astype(np.float32), device=device)
    obs, actions, next_obs = normalizer.unpack(decoded, mixed.obs_dim, mixed.action_dim)
    actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)

    finite = (np.isfinite(obs).all(1) & np.isfinite(actions).all(1)
              & np.isfinite(next_obs).all(1) & np.isfinite(r_new))
    starts_terminal = terminal_flags(task, obs)
    keep = finite & ~starts_terminal
    synthetic = Transitions(obs[keep], actions[keep], r_new[keep], next_obs[keep],
                            terminals=terminal_flags(task, next_obs[keep]))
    stats = {"anchors": float(len(anchors)), "dropped_nonfinite": float((~finite).sum()),
             "dropped_initial_terminal": float((finite & starts_terminal).sum()),
             "kept": float(len(synthetic)), "target_reward_mean": float(target.rewards.mean())}
    if len(synthetic):
        stats["synthetic_reward_mean"] = float(synthetic.rewards.mean())
        stats["synthetic_terminal_rate"] = float(synthetic.terminals.mean())
    logger("      kept {}/{} synthetic transitions".format(int(stats["kept"]), len(anchors)))
    return AugmentationArtifacts(synthetic=synthetic, vae=model, normalizer=normalizer, field=field,
                                 frame_source=frame_source, frame_target=frame_target, stats=stats)
