"""Run one cell"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rift.experiment import RunConfig, run  # noqa: E402

NESTED = {"iql", "vae", "transport"}
BOOL_WORDS = {"true": True, "false": False, "yes": True, "no": False}


def coerce(text: str):
    if text.strip().lower() in BOOL_WORDS:
        return BOOL_WORDS[text.strip().lower()]
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(payload: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise SystemExit("--set expects key=value, got {!r}".format(item))
        key, _, value = item.partition("=")
        parsed = coerce(value)
        if "." in key:
            section, _, leaf = key.partition(".")
            if section not in NESTED:
                raise SystemExit("unknown config section {!r}".format(section))
            target, leaf_key = payload.setdefault(section, {}), leaf
        else:
            target, leaf_key = payload, key
        current = target.get(leaf_key)
        if isinstance(current, bool) and not isinstance(parsed, bool):
            raise SystemExit("{} is a boolean; got {!r} (use true/false)".format(key, value))
        target[leaf_key] = parsed
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--skip-existing", action="store_true",
                    help="exit without running if this run's result JSON already exists")
    args = ap.parse_args(argv)

    payload: dict = {}
    if args.config is not None:
        payload = yaml.safe_load(args.config.read_text()) or {}
    payload = apply_overrides(payload, args.set)
    unknown = set(payload) - {f.name for f in dataclasses.fields(RunConfig)}
    if unknown:
        raise SystemExit("unknown config keys: {}".format(sorted(unknown)))
    config = RunConfig(**payload)
    result = pathlib.Path(config.output_dir) / (config.run_name + ".json")
    if args.skip_existing and result.exists():
        print("skipping {}: {} exists".format(config.run_name, result))
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
