"""ODRL shifted locomotion environments on modern MuJoCo;

ODRL builds its target domains from modified MJCF files loaded through
``gym.envs.mujoco.*_v3`` (mujoco-py), which does not install on current
Python. The assets in ``assets/`` are ODRL's files converted to the gymnasium
``-v4`` format; ``-v4`` rather than ``-v5`` because v5 changed walker2d's
right-foot friction, and every D4RL and ODRL transition was generated under
the v3/v4 geometry.
"""
from __future__ import annotations

import os
import pathlib
from typing import Callable

import numpy as np

TASKS: tuple[str, ...] = ("halfcheetah", "hopper", "walker2d", "ant")
BASE_ASSET: dict[str, str] = {"halfcheetah": "half_cheetah.xml", "hopper": "hopper.xml",
                              "walker2d": "walker2d.xml", "ant": "ant.xml"}
CONTINUOUS_LEVELS: tuple[float, ...] = (0.1, 0.5, 2.0, 5.0)
DISCRETE_LEVELS: tuple[str, ...] = ("easy", "medium", "hard")
DEFAULT_ASSET_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets"


def _env_class(task: str):
    if task == "halfcheetah":
        from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
        return HalfCheetahEnv
    if task == "hopper":
        from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
        return HopperEnv
    if task == "walker2d":
        from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
        return Walker2dEnv
    if task == "ant":
        from gymnasium.envs.mujoco.ant_v4 import AntEnv
        return AntEnv
    raise ValueError("unknown task {!r}; expected one of {}".format(task, TASKS))


def asset_name(task: str, shift: str | None = None, level: float | str | None = None) -> str:
    if task not in TASKS:
        raise ValueError("unknown task {!r}".format(task))
    if shift is None:
        return BASE_ASSET[task]
    if shift in ("friction", "gravity"):
        if level is None or float(level) not in CONTINUOUS_LEVELS:
            raise ValueError("{} shift needs a level in {}, got {!r}".format(shift, CONTINUOUS_LEVELS, level))
        return "{}_{}_{}.xml".format(task, shift, float(level))
    if shift.startswith("morph"):
        if level not in DISCRETE_LEVELS:
            raise ValueError("{} shift needs a level in {}, got {!r}".format(shift, DISCRETE_LEVELS, level))
        return "{}_{}_{}.xml".format(task, shift, level)
    raise ValueError("unknown shift family {!r}".format(shift))


def make_env(task: str, shift: str | None = None, level: float | str | None = None,
             xml_path: str | os.PathLike | None = None, asset_dir: str | os.PathLike | None = None,
             **kwargs):
    import gymnasium.envs.mujoco.mujoco_env as mujoco_env

    cls = _env_class(task)
    if task == "ant":
        kwargs.setdefault("use_contact_forces", False)
    if xml_path is None and shift is not None:
        root = pathlib.Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
        xml_path = root / asset_name(task, shift, level)
    if xml_path is None:
        return cls(**kwargs)
    xml_path = pathlib.Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError("asset {} not found".format(xml_path))

    absolute = str(xml_path.resolve())
    original = mujoco_env.MujocoEnv.__init__

    def redirected(self, model_path, frame_skip, **inner):
        original(self, absolute, frame_skip, **inner)

    mujoco_env.MujocoEnv.__init__ = redirected
    try:
        return cls(**kwargs)
    finally:
        mujoco_env.MujocoEnv.__init__ = original


def _halfcheetah_terminal(obs: np.ndarray) -> np.ndarray:
    return np.zeros(len(obs), dtype=bool)


def _hopper_terminal(obs: np.ndarray) -> np.ndarray:
    z, angle, state = obs[:, 0], obs[:, 1], obs[:, 1:]
    healthy = np.all((state > -100.0) & (state < 100.0), axis=1) & (z > 0.7) & (np.abs(angle) < 0.2)
    return ~healthy


def _walker2d_terminal(obs: np.ndarray) -> np.ndarray:
    z, angle = obs[:, 0], obs[:, 1]
    return ~((z > 0.8) & (z < 2.0) & (angle > -1.0) & (angle < 1.0))


def _ant_terminal(obs: np.ndarray) -> np.ndarray:
    z = obs[:, 0]
    return ~(np.isfinite(obs).all(axis=1) & (z >= 0.2) & (z <= 1.0))


TERMINATION_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "halfcheetah": _halfcheetah_terminal, "hopper": _hopper_terminal,
    "walker2d": _walker2d_terminal, "ant": _ant_terminal}


def terminal_flags(task: str, next_obs: np.ndarray) -> np.ndarray:
    next_obs = np.atleast_2d(np.asarray(next_obs, dtype=np.float64))
    return TERMINATION_FNS[task](next_obs)
