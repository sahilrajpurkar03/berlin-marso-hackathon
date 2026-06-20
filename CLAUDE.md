# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Starter kit for the **Marso Hack Berlin 2026** robot-learning hackathon: train a Franka Panda arm
(ManiSkill 3, GPU sim) to sort parcels into color-matched bins from a fixed scene-camera RGB image
+ proprioception only — **no privileged parcel/bin state at eval time**. Scoring is `sort_accuracy`
(fraction of parcels in the correct-color bin) on held-out configs, weighted
`0.2·easy + 0.3·medium + 0.5·hard`. The provided RGB Diffusion Policy is a runnable **template that
does not solve the task out of the box** — making it (or another approach) actually sort is the
challenge. Submitting a scripted/privileged-state policy is grounds for disqualification.

## Commands

```bash
# setup
pixi install
pixi run install                              # pip install -e .

# fetch the 200-demos-per-level rgb dataset (Kaggle data; auto-mounted on Kaggle)
pixi run python il/download_demos.py

# train the RGB Diffusion Policy (Hydra dispatcher -> il/baselines/diffusion_policy/train_rgbd.py)
pixi run python il/train.py method=dp_rgb demo_dir=easy
pixi run python il/train.py method=dp_rgb demo_dir=easy flags.total_iters=8000 flags.eval_freq=4000

# evaluate a checkpoint — IDENTICAL interface to judging (only eval_config differs)
pixi run python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_sort_accuracy.pt \
    eval_config=conf/eval/default.yaml
# swap difficulty=easy|medium|hard to check all three before estimating the weighted final_score

# optional: record more demos (scripted policy; record -> replay -> media)
pixi run python il/gen_demos.py --difficulty easy --num-episodes 200
```

There is no test suite (`pixi run test` points at a `test.py` that does not exist in this checkout
— don't rely on it).

## Architecture — how the pieces connect

```
demos (il/demos/<level>/*.h5+.json)
        │
        ▼
il/train.py  (Hydra dispatcher; NOT the learning code itself)
   loads il/conf/method/<method>.yaml, converts its `flags:` dict to the vendored
   baseline's tyro CLI (underscore->--hyphen, bool->--flag/--no-flag), then
   subprocess-runs the real trainer (il/baselines/diffusion_policy/train_rgbd.py)
        │
        ▼
checkpoint.pt  (il/baselines/<baseline_dir>/runs/<exp_name>/checkpoints/*.pt
                — dict with "agent" and "ema_agent"; eval uses ema_agent)
        │
        ▼
warehouse_sort/il_policy.py: load_dp_rgb(checkpoint, sample_obs, action_space, device)
   rebuilds the Agent class from train_rgbd.py (imported by sys.path hack into
   il/baselines/diffusion_policy) and wraps it in a class exposing .act(obs)
        │
        ▼
policy.act(obs, deterministic=True) -> Tensor (num_envs, 4) in [-1, 1]
        │
        ▼
eval.py -> warehouse_sort/utils.py: rollout_metrics() steps the env and computes
   sort_accuracy from WarehouseSort-v1's evaluate() (geometric, deterministic)
```

**The one contract that matters for any custom approach** (see SUBMISSION.md §3):
`load_fn(checkpoint, sample_obs, action_space, device) -> policy` where
`policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]`.
`eval.py`/the judge only ever calls `load_fn` and `.act(obs)` — it never imports training code,
so a custom policy module must be importable standalone. Build the model from `sample_obs` shapes
(`sample_obs["rgb"]`, `sample_obs["state"]`) rather than hardcoding dims; read the checkpoint
format yourself (only your `load_fn` reads it).

### Key files

| File | Role |
|------|------|
| `warehouse_sort/env.py` | `WarehouseSort-v1` ManiSkill env: scene/parcels/bins, obs construction, sparse reward, `evaluate()` (the success check) |
| `warehouse_sort/il_policy.py` | `load_dp_rgb` — reference policy entrypoint; template for custom `load_fn`s |
| `warehouse_sort/utils.py` | `make_env`, `load_agent`, `rollout_metrics` (the exact judge rollout loop), `record_eval_video` |
| `eval.py` | Evaluates a checkpoint on an `eval_config` — same code path used for judging; only the config (seeds/randomization) differs |
| `il/train.py` | Hydra dispatcher only — has no learning logic itself |
| `il/baselines/diffusion_policy/train_rgbd.py` | The actual RGB Diffusion Policy trainer (vendored from ManiSkill example baselines) |
| `examples/scripted_policy.py` | Privileged-state waypoint controller used only to generate demos — never a valid submission |
| `conf/` | Hydra configs: `config.yaml` (root), `difficulty/{easy,medium,hard}.yaml`, `eval/default.yaml` |
| `submission.yaml` (you create this) | Manifest judges read: `policy:` entrypoint + per-level `checkpoint:` paths |

### Observation / action / reward contract (fixed across all difficulties)

- `obs["rgb"]`: `(N, 128, 128, 3)` uint8 from one fixed third-person scene camera
- `obs["state"]`: `(N, 26)` float32 proprioception only (TCP pose, gripper, joints) — **no** parcel/bin/tag info
- Action: `pd_ee_delta_pos`, 4 dims in `[-1, 1]` — `[0:3]` EE delta xyz (±0.1 m/step), `[3]` gripper (+1 open / −1 close)
- Reward: sparse `+1` per newly-correctly-placed parcel; `compute_dense_reward` is unimplemented (raises) — any RL approach must design its own shaping
- Success check (`WarehouseSort-v1.evaluate()` in `env.py`): a parcel counts once it's inside the
  matching-color bin's xy footprint, below the rim height, and released (ungrasped)

### Difficulty levels (`conf/difficulty/*.yaml`)

| level | parcels | randomization |
|---|---|---|
| easy | 2 | fully fixed positions, bins never swap |
| medium | 4 | parcel xy jitter ±0.015 m, fixed orientation, bins fixed |
| hard | 6 | xy jitter ±0.02 m, yaw jitter ±0.1 rad, bins swap sides ~50% of episodes |

Because the rgb obs shape is identical at every difficulty, **one checkpoint can be evaluated
across all three levels** — but the held-out judge configs widen position randomization further, so
training only on `easy` will not generalize to `hard` (which carries half the final score weight).

### Diffusion Policy template specifics (`il/conf/method/dp_rgb.yaml`)

ResNet18 + SpatialSoftmax (32 keypoints) visual encoder → 256-d features, concatenated with
proprioception, FiLM-conditioned ConditionalUnet1D diffusion head. `obs_horizon=2`, `act_horizon=8`,
`pred_horizon=16`, 100 DDPM train steps / 16 inference steps by default, `total_iters=30000`,
`batch_size=128`. If you change an architecture/horizon flag for training (`obs_horizon`,
`act_horizon`, `pred_horizon`, `unet_dims`, `diffusion_step_embed_dim`, `n_groups`,
`visual_encoder`, `num_kp`), you must pass the matching value into `load_dp_rgb(...)` or the
checkpoint won't load — those args aren't persisted in the checkpoint itself.
