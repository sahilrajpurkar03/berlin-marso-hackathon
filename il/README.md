# Imitation Learning — WarehouseSort

Main track is **state-based**. The IL pipeline follows ManiSkill 3's standard approach:
**demos → train state Diffusion Policy → evaluate via eval.py**. A state DP runs end-to-end as a
starting point. An optional **RGB** image track is also provided as a template
(`method=dp_rgb`, `load_dp_rgb`) — harder, not yet solving the task.

---

## Step 1 — Demonstrations (provided)

**You don't need to record anything.** We provide pre-recorded demos for every level
(**200 episodes per level**, `state` for the main track + `rgb` for the image track) as the
[Kaggle competition
data](https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge/data).

- **On Kaggle**: the competition data is mounted under `/kaggle/input/` automatically.
- **Elsewhere**: fetch it once (join the competition + set a Kaggle API token first):

```bash
pixi run python il/download_demos.py
```

Either way the files are staged into `il/demos/<level>/`. **Each dataset is a pair** — the
`trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5` (data) **and** the matching
`trajectory.rgb.pd_ee_delta_pos.physx_cuda.json` (control-mode metadata). Both are required and
must sit together; the trainer finds the `.json` next to the `.h5`.

> ⚠️ The demos are recorded by rolling out a **scripted waypoint policy**
> (`examples/scripted_policy.py`). Using it to *collect data* is exactly how we built these
> datasets and is fine. **Submitting a scripted / hard-coded controller is not allowed and
> leads to disqualification** — your submitted policy must act from the observation.

### Optional — generate more demos

If you want extra data, record more with the same tool. Demos are clean scripted trajectories
(no action noise) and each episode ends the moment all parcels are sorted:

```bash
pixi run python il/gen_demos.py --difficulty easy   --num-episodes 200
pixi run python il/gen_demos.py --difficulty medium --num-episodes 200
pixi run python il/gen_demos.py --difficulty hard   --num-episodes 200
```

**How it works — record → replay → media** (the standard ManiSkill data pipeline):

1. **Record** — rolls the scripted waypoint policy across `--num-episodes` seeds (one env at a
   time) and writes the raw trajectories + env states to `il/demos/<level>/trajectory.h5` via
   ManiSkill's `RecordEpisode`. (The scripted policy reads privileged sim state to *control* the
   arm; this is only the data generator, not a submittable policy.)
2. **Replay** — runs ManiSkill's `replay_trajectory` to re-execute the recorded actions and
   render fresh **rgb** observations, producing the training-ready dataset the trainer loads:
   `trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5`. Replay runs single-env (GPU) because it
   reproduces each trajectory from recorded env states.
3. **Media** — saves a demo `media/<level>_demo.mp4` and `media/<level>_demo.gif`
   (render + sensor views) so you can eyeball what a successful episode looks like.

Useful flags: `--no-replay` (raw demos only), `--no-media` (skip the mp4/gif), `--base-seed`
(shift the seed range so new demos differ from the provided set). Run
`pixi run python il/gen_demos.py --help` for all options.

---

## Step 2 — Train

```bash
pixi run python il/train.py method=dp demo_dir=easy        # state Diffusion Policy (main track)
```

Train each level separately (state is parcel-count-specific → **one checkpoint per level**):
```bash
pixi run python il/train.py method=dp demo_dir=medium
pixi run python il/train.py method=dp demo_dir=hard
```

Override any hyperparameter on the CLI (e.g. a longer prediction horizon):
```bash
pixi run python il/train.py method=dp flags.total_iters=50000 flags.pred_horizon=32
```

Checkpoints land at `il/baselines/diffusion_policy/runs/<exp_name>/checkpoints/` (the state
method's default `exp_name` is `warehouse_state_dp_easy`). For the optional image track use
`method=dp_rgb`.

**Datasets in a custom location** (e.g. the mounted Kaggle competition data, not `il/demos/`): pass `demo_path=`
the full path to the `.h5` — keep the matching `.json` in the same folder:

```bash
pixi run python il/train.py method=dp \
    demo_path=/kaggle/input/<competition>/easy/trajectory.state.pd_ee_delta_pos.physx_cuda.h5
```

---

## Step 3 — Evaluate

```bash
pixi run python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_state_dp_easy/checkpoints/best_eval_sort_accuracy.pt \
    eval_config=conf/eval/default.yaml
```

Use the matching per-level checkpoint for `difficulty=medium` / `difficulty=hard`.

---

## Training time

Rough training time for the provided **state** Diffusion Policy at default settings (single modern
GPU, e.g. Colab T4). The optional image (rgb) track is a template that does not yet solve the task.

| method | obs | level | default iters | approx. train time |
|--------|-----|-------|:---:|:---:|
| DP | state | easy   | 30k | ~20–40 min |
| DP | state | medium | 50k | ~40–70 min |
| DP | state | hard   | 60k | ~50–90 min |
| DP | rgb (scene) | any | 30k | ~30–90 min |

(Training time is for a single modern GPU, e.g. Colab T4.)

---

## Technical notes

- **Why Diffusion Policy?** A plain MLP behavior cloner collapses here due to compounding error;
  Diffusion Policy's action chunking helps. (It's still only a starting point — the image
  template does not yet solve the task.)
- **Image input = a single fixed third-person scene camera.** It keeps the whole workspace
  (robot + parcels + bins) in frame the entire episode and has the same shape at any parcel
  count, so one policy can run across difficulties.
- ManiSkill 3.0.1 pip wheel does not ship `examples/baselines`, so the DP baseline is
  vendored in `il/baselines/diffusion_policy/`.
- Set `HDF5_USE_FILE_LOCKING=FALSE` if replay/load races on the just-written `.h5`.
