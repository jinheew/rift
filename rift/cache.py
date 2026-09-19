"""cache for the fitted VAE and improvement field"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Callable

import torch


def fingerprint(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


@dataclasses.dataclass
class ArtifactCache:
    root: pathlib.Path
    cell: str
    hits: dict = dataclasses.field(default_factory=dict)     # stage -> seconds the artifact cost

    @property
    def directory(self) -> pathlib.Path:
        return pathlib.Path(self.root) / self.cell

    def path(self, stage: str, config: Any) -> pathlib.Path:
        return self.directory / "{}_{}.pt".format(stage, fingerprint(config))

    def load(self, stage: str, config: Any) -> Any | None:
        path = self.path(stage, config)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:    
            return None
        self.hits[stage] = float(payload.get("compute_seconds") or 0.0)
        return payload["artifact"]

    def save(self, stage: str, config: Any, artifact: Any,
             compute_seconds: float | None = None) -> pathlib.Path:
        path = self.path(stage, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp{}".format(os.getpid()))
        torch.save({"artifact": artifact, "compute_seconds": compute_seconds}, tmp)
        os.replace(tmp, path)
        return path

    def get_or_compute(self, stage: str, config: Any, compute: Callable[[], Any], logger=print) -> Any:
        cached = self.load(stage, config)
        if cached is not None:
            logger("  cache hit: {} ({})".format(stage, self.path(stage, config)))
            return cached
        started = time.time()
        artifact = compute()
        self.save(stage, config, artifact, compute_seconds=time.time() - started)
        return artifact
