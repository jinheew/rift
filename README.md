<!-- # RIFT: Reward Improvement Field Transfer

Data augmentation for cross-domain offline reinforcement learning. Given a
large source dataset and a small target dataset collected under different
dynamics, RIFT learns in the source how low-reward transitions relate to
high-reward ones, applies that relation to the target's low-reward
transitions, and hands the augmented target dataset to an unmodified offline
RL learner (IQL).

The three stages, each in its own module:

| stage | what it does | module |
|---|---|---|
| 1 | a VAE over transitions `(s, a, s')`, shared by both contexts, with balanced batches and a class-conditional prior on one latent coordinate | `rift/method/vae.py` |
| 2 | each context's `(z, r)` points standardised; in the source, mini-batch optimal transport between the bottom-`q` and top-`q` reward classes, and an affine field `delta(u) = W u + b` fitted to the matched pairs | `rift/method/transport.py` |
| 3 | the target's low-reward transitions moved along the field in the target's coordinates, decoded, rewarded and given terminal flags | `rift/method/augment.py` |
| 4 | the augmented dataset expressed in standardised observation coordinates (the stage-1 scaler), which the policy's inputs also go through at run time | `rift/method/augment.py` |

The learner (`rift/learner/iql.py`) is IQL as implemented in the ODRL
benchmark (the code base OTDF and STC also build on), unmodified: it trains
on the dataset RIFT hands it, either the target data plus the synthetic
transitions or source + target + synthetic with half-source / half-target
batches. The paper reports the better of the two training sets per cell.

## Install

Python 3.10+, CPU is enough (a run is a few CPU-hours).

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest tests -q       # 16 tests, no data needed
```

## Data

* **Target** (context B): the ODRL shifted-domain datasets, 5,000 transitions
  each. The 24 files the paper uses are in `data/target/` (gravity 0.5,
  friction 0.5 and the morphology variants, medium and expert, for ant,
  halfcheetah, hopper and walker2d). Any other ODRL file is fetched from ODRL's
  Google Drive folder on first use.
* **Source** (context A): the D4RL `-v2` MuJoCo datasets (medium,
  medium-replay, medium-expert), downloaded on first use from the
  `imone/D4RL` HuggingFace mirror into `data/source/` (300 to 700 MB each).
* **Environments**: ODRL's modified MJCF files, converted to gymnasium's
  `-v4` format, in `assets/`. Nothing else is needed: no `d4rl`, no
  `mujoco-py`.

## Run

One cell, target-only arm, seed 0:

```bash
python scripts/run.py --config configs/rift.yaml --set task=hopper shift=gravity level=0.5 \
    source_quality=medium target_quality=medium seed=0 include_source=false
```

The source-included arm of the same cell is `include_source=true`. Morphology
cells use `shift=morph_halflegs|morph_thigh|morph_foot|morph_leg level=medium`
(ant, halfcheetah, hopper, walker2d), friction cells `shift=friction level=0.5`.
Each run writes `results/rift/<run name>.json` with the configuration, the
evaluation every 10k steps (10 episodes) and the final score; the fitted VAE
and field of a cell are cached under `results/cache/` and shared by its two
arms. A run that is interrupted resumes from its last checkpoint when the same
command is rerun.

A quick end-to-end check at toy scale (about a minute):

```bash
python scripts/run.py --config configs/smoke.yaml
```

### The full evaluation

The 48 cells of the paper (STC's Tables 1, 2 and 6), five seeds, both arms:

```bash
python scripts/make_run_list.py                 # runs.txt: 480 lines
xargs -P 8 -I{} sh -c "{}" < runs.txt           # or a Slurm array over the lines
python scripts/summarize.py --root results/rift # per-cell mean +- std and the selected arm
```

Each line is an ordinary `scripts/run.py` invocation with `--skip-existing`,
so a list can be resubmitted until every result exists.

### Ablations

Everything stays at the protocol setting except the one key changed:

```bash
python scripts/make_run_list.py --blocks gravity morph --extra transport.quantile=0.1 --root results/q0.1
python scripts/make_run_list.py --blocks gravity morph --extra vae.variant=bal     --root results/vae_bal
python scripts/make_run_list.py --blocks gravity morph --extra vae.variant=base    --root results/vae_base
```

`transport.quantile` is the reward quantile `q` (Table 3); `vae.variant` is
the regulariser (Table 4: `base`, `bal`, `bal+1p`).

## Hyperparameters

All in `configs/rift.yaml`, fixed across every cell:

| | value |
|---|---|
| reward quantile `q` | 0.2 |
| VAE | latent 16, hidden 256 x 2, beta 0.005, 100k steps, lr 1e-3, batch 256 |
| VAE regulariser | half of each batch from the target; class prior N(±c_q, s_q²) on z_0 with weight 0.5, every 2 steps |
| optimal transport | exact, 8 problems of 2,000 x 2,000 points, squared Euclidean cost on the standardised latent |
| field | affine, ordinary least squares, ridge 1e-6 |
| anchors | the target's bottom-`q` transitions; one synthetic transition each |
| IQL | ODRL implementation: expectile 0.7, beta 3.0, discount 0.99, lr 3e-4, 256 x 2 networks, 1M steps, batch 256 (128 + 128 with the source) |
| stage 4 | observations standardised per dimension with source + target statistics; ODRL, OTDF and STC train on raw observations |
| evaluation | 10 episodes every 10k steps, final score reported, 5 seeds |

The VAE trains on all source rows but 5,000 and on 90% of the target rows,
drawn with the run's seed; the transport, the anchors and the learner use every
row. This reproduces the paper's runs, whose held-out slices served the VAE
diagnostics. Set `vae.holdout_target_fraction=0 vae.holdout_source=0` to train
on everything.

## Layout

```
rift/data        dataset loading (D4RL source, ODRL target) and download
rift/envs        ODRL environments on gymnasium, termination rules, reference scores
rift/method      the three stages of RIFT
rift/learner     IQL and target-domain evaluation
rift/experiment  one cell end to end
scripts/         run.py, make_run_list.py, summarize.py
configs/         rift.yaml (the protocol), smoke.yaml
assets/          ODRL MJCF files (gymnasium v4 format)
data/target      ODRL target datasets used in the paper
``` -->
