"""Fetch the source (D4RL) and target (ODRL) offline datasets"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import urllib.request

# HuggingFace mirror of the original D4RL HDF5 files.
D4RL_REPO = "imone/D4RL"
D4RL_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"

# ODRL's target-domain dataset folder.
ODRL_FOLDER_ID = "1fwkjtXCbMxVP7RM7NSN3mF40Gakwkqei"
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data"

# D4RL dataset qualities used as source domains.
SOURCE_QUALITIES = ("medium", "medium_replay", "medium_expert", "expert", "random")


def d4rl_filename(task: str, quality: str) -> str:
    if quality not in SOURCE_QUALITIES:
        raise ValueError("unknown source quality {!r}; expected one of {}".format(
            quality, SOURCE_QUALITIES))
    return "{}_{}-v2.hdf5".format(task, quality)


def odrl_filename(task: str, shift: str, level: float | str, quality: str) -> str:
    if shift in ("friction", "gravity"):
        level = float(level)
    return "{}_{}_{}_{}.hdf5".format(task, shift, level, quality)


def _download(url: str, dest: pathlib.Path, desc: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rift-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print("\r  {} {:5.1f}% ({:.1f} MB)".format(desc, pct, done / 1e6),
                      end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    tmp.replace(dest)


def fetch_source(task: str, quality: str, root: pathlib.Path | str = DEFAULT_ROOT,
                 force: bool = False) -> pathlib.Path:
    root = pathlib.Path(root)
    name = d4rl_filename(task, quality)
    dest = root / "source" / name
    if dest.exists() and not force:
        return dest
    _download(D4RL_URL.format(repo=D4RL_REPO, name=name), dest, name)
    return dest


_ODRL_INDEX: dict[str, str] | None = None


def odrl_index(root: pathlib.Path | str = DEFAULT_ROOT,
               refresh: bool = False) -> dict[str, str]:
    global _ODRL_INDEX
    root = pathlib.Path(root)
    cache = root / "target" / "odrl_index.json"
    if _ODRL_INDEX is not None and not refresh:
        return _ODRL_INDEX
    if cache.exists() and not refresh:
        _ODRL_INDEX = json.loads(cache.read_text())
        return _ODRL_INDEX

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError(
            "gdown is required to resolve the ODRL dataset folder "
            "(pip install -r requirements.txt)"
        ) from exc

    print("resolving ODRL Google Drive folder (slow, cached afterwards)...", file=sys.stderr)
    entries = gdown.download_folder(
        id=ODRL_FOLDER_ID, skip_download=True, quiet=True, use_cookies=False
    )
    if not entries:
        raise RuntimeError(
            "could not list the ODRL Drive folder. Download it manually from\n"
            "  https://drive.google.com/drive/folders/{}\n"
            "and place the locomotion .hdf5 files in {}".format(
                ODRL_FOLDER_ID, root / "target")
        )
    index = {pathlib.PurePath(e.path).name: e.id for e in entries}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index, indent=1, sort_keys=True))
    _ODRL_INDEX = index
    return index


def fetch_target(task: str, shift: str, level: float | str, quality: str,
                 root: pathlib.Path | str = DEFAULT_ROOT,
                 force: bool = False) -> pathlib.Path:
    root = pathlib.Path(root)
    name = odrl_filename(task, shift, level, quality)
    dest = root / "target" / name
    if dest.exists() and not force:
        return dest

    # Accept a manual download placed anywhere under data/target.
    for candidate in (root / "target").rglob(name):
        if candidate.is_file():
            if candidate != dest:
                shutil.copy2(candidate, dest)
            return dest

    import gdown

    index = odrl_index(root)
    if name not in index:
        raise FileNotFoundError(
            "{} is not in the ODRL Drive folder. Available names look like "
            "'hopper_gravity_2.0_medium.hdf5'.".format(name)
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("  fetching {}".format(name), file=sys.stderr)
    gdown.download(id=index[name], output=str(dest), quiet=True)
    if not dest.exists():
        raise RuntimeError(
            "gdown failed for {}. Download it manually from\n"
            "  https://drive.google.com/drive/folders/{}\n"
            "and place it at {}".format(name, ODRL_FOLDER_ID, dest)
        )
    return dest
