"""Write the run commands of the paper's evaluation for slurm:

    python scripts/make_run_list.py                          # 48 cells x 5 seeds x 2 arms = 480 runs
    python scripts/make_run_list.py --blocks gravity morph   # Tables 1 and 2 only
    python scripts/make_run_list.py --extra transport.quantile=0.1 --root results/q0.1   # an ablation
"""
from __future__ import annotations

import argparse
import pathlib
import sys

TASKS = ("ant", "halfcheetah", "hopper", "walker2d")
MORPH = {"ant": "morph_halflegs", "halfcheetah": "morph_thigh", "hopper": "morph_foot", "walker2d": "morph_leg"}
BLOCKS = {
    "gravity": [("medium", "medium"), ("medium", "medium_expert"), ("medium", "expert"),
                ("medium_replay", "medium"), ("medium_expert", "medium")],
    "morph": [("medium", "medium"), ("medium", "expert"), ("medium_replay", "medium"), ("medium_expert", "medium")],
    "friction": [("medium", "medium"), ("medium", "medium_expert"), ("medium", "expert")],
}
LINE = ("python scripts/run.py --config configs/rift.yaml --skip-existing --set task={task} shift={shift} "
        "level={level} source_quality={src} target_quality={tgt} seed={seed} include_source={source} "
        "output_dir={root}/{block}/{arm}{extra}")


def shift_level(block: str, task: str) -> tuple[str, str]:
    return (MORPH[task], "medium") if block == "morph" else (block, "0.5")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs.txt"))
    ap.add_argument("--root", default="results/rift")
    ap.add_argument("--blocks", nargs="+", default=list(BLOCKS), choices=list(BLOCKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=["B", "AB"], choices=("B", "AB"))
    ap.add_argument("--extra", nargs="*", default=[], metavar="KEY=VALUE",
                    help="appended to every line's --set (ablations: transport.quantile=..., vae.variant=...)")
    args = ap.parse_args(argv)
    extra = "".join(" " + item for item in args.extra)
    lines = []
    for seed in args.seeds:
        for block in args.blocks:
            for src, tgt in BLOCKS[block]:
                for task in TASKS:
                    shift, level = shift_level(block, task)
                    for arm in args.arms:
                        lines.append(LINE.format(task=task, shift=shift, level=level, src=src, tgt=tgt, seed=seed,
                                                 source=str(arm == "AB").lower(), root=args.root, block=block,
                                                 arm=arm, extra=extra))
    args.out.write_text("".join(line + "\n" for line in lines), newline="\n")
    print("wrote {} ({} runs)".format(args.out, len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
