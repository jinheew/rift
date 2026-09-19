# RIFT: Reward Improvement Field Transfer

RIFT is a data-augmentation method for **cross-domain offline reinforcement learning**.

Given a large **source** dataset and a small **target** dataset collected under different dynamics, RIFT learns how low-reward source transitions relate to high-reward ones, transfers that relation to the target domain, and augments the target dataset before training a standard offline RL learner.

The offline learner is **IQL**, used without modification.

## Method

RIFT has four steps:

| Stage | Description | Module |
|---|---|---|
| 1 | Train a shared VAE over transitions `(s, a, s')` using balanced source/target batches and a class-conditional prior on one latent coordinate. | `rift/method/vae.py` |
| 2 | Standardise `(z, r)` within each domain, match low- and high-reward source transitions with optimal transport, and fit an affine field `delta(u) = Wu + b`. | `rift/method/transport.py` |
| 3 | Apply the learned field to low-reward target transitions, decode them, and assign rewards and terminal flags. | `rift/method/augment.py` |
| 4 | Express the final dataset in the same standardised observation coordinates used by the policy at evaluation time. | `rift/method/augment.py` |

IQL then trains on either:

- target + synthetic transitions, or
- source + target + synthetic transitions, using half-source / half-target batches.

The paper reports the better arm for each evaluation cell.

## Install

Requires Python 3.10+. CPU execution is sufficient.

```bash
python -m venv .venv
. .venv/bin/activate    

pip install -r requirements.txt
pip install -e . --no-deps
```

## Data

**Target datasets.** ODRL shifted-domain datasets with 5,000 transitions each. The 24 datasets used in the paper are included in `data/target/`. Other ODRL datasets are downloaded on first use.
**Source datasets.** D4RL `-v2` MuJoCo datasets (`medium`, `medium-replay`, `medium-expert`), downloaded from the `imone/D4RL` HuggingFace mirror into `data/source/`.
**Environments.** ODRL modified MuJoCo environments are included under `assets/` in Gymnasium `-v4` format. 

## Run

Example: Hopper, gravity shift, target-only arm, seed 0.

```bash
python scripts/run.py \
  --config configs/rift.yaml \
  --set task=hopper shift=gravity level=0.5 \
  source_quality=medium target_quality=medium \
  seed=0 include_source=false
```

Use `include_source=true` for the source-included arm.

Other shifts:

```text
shift=friction level=0.5

shift=morph_halflegs
shift=morph_thigh
shift=morph_foot
shift=morph_leg
```

## Ablations

Example:

```bash
python scripts/make_run_list.py \
  --blocks gravity morph \
  --extra transport.quantile=0.1 \
  --root results/q0.1
```

`transport.quantile` controls the reward quantile `q`.

`vae.variant` selects the VAE regulariser: `base`, `bal`, or `bal+1p`.

## Main hyperparameters

| Setting | Value |
|---|---|
| Reward quantile | `q = 0.2` |
| VAE | latent 16, hidden 256×2, beta 0.005, 100k steps, lr 1e-3, batch 256 |
| VAE regulariser | 50% target batches; class prior on `z_0`, weight 0.5 |
| Optimal transport | exact, 8 × 2,000-by-2,000 problems |
| Transport cost | squared Euclidean distance in standardised latent space |
| Field | affine OLS, ridge `1e-6` |
| Synthetic anchors | target bottom-`q` transitions, one synthetic transition each |
| IQL | expectile 0.7, beta 3.0, discount 0.99, lr 3e-4, hidden 256×2 |
| IQL training | 1M steps, batch 256 |
| Evaluation | 10 episodes every 10k steps, 5 seeds |

To train the VAE on all available data:

```bash
--set vae.holdout_target_fraction=0 vae.holdout_source=0
```

## Repository layout

```text
rift/data        dataset loading and downloads
rift/envs        ODRL environments and reference scores
rift/method      VAE, transport, and augmentation
rift/learner     IQL and evaluation
rift/experiment  end-to-end experiment logic

scripts/         run and evaluation scripts
configs/         experiment configurations
assets/          MuJoCo XML assets
data/target      target datasets used in the paper
```

