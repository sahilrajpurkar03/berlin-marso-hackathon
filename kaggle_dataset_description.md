# WarehouseSort — Demonstration Datasets

Expert demonstrations for the **WarehouseSort** pick-and-place challenge: a Franka Panda robot
sorts parcels into the bin matching the **colored tag on each parcel's top face** (red tag → red
bin, blue tag → blue bin).

## What's included

**200 demonstration episodes per difficulty level**, organized by folder:

| folder | level | parcels | randomization |
|--------|-------|---------|---------------|
| `easy/`   | easy   | 2 | fully fixed layout |
| `medium/` | medium | 4 | small position jitter |
| `hard/`   | hard   | 6 | small position + slight orientation jitter; bins may swap sides |

Each folder contains ManiSkill trajectory datasets (each is a **pair** — `.h5` + `.json`, both
required):

- `trajectory.state.pd_ee_delta_pos.physx_cuda.{h5,json}` — **state** track (main)
- `trajectory.rgb.pd_ee_delta_pos.physx_cuda.{h5,json}` — **rgb** track (optional image track)

**Observations.** *State* (main track): a low-dim vector with robot proprioception + parcel poses
& tag colors + bin positions & colors (its length grows with the parcel count, so it is
level-specific). *RGB* (optional): a fixed third-person scene-camera image `(128, 128, 3)` uint8 +
proprioception `(26,)`. **Action** (both): `pd_ee_delta_pos`, 4 dims in `[-1, 1]` (end-effector
Δxyz + gripper).

## How they were generated (ManiSkill 3)

A deterministic **scripted waypoint policy** solves the task in the GPU-accelerated
[ManiSkill 3](https://maniskill.readthedocs.io/en/latest/) simulator. We **record** each rollout
with ManiSkill's `RecordEpisode`, then **replay** it (`replay_trajectory`) to render the
observations in each mode — the standard ManiSkill *record → replay* pipeline. Trajectories are
clean (no action noise) and each episode ends the moment all parcels are correctly sorted.

> The scripted policy reads privileged simulator state to drive the arm — it is the **data
> generator only**. Submitted policies must act from the observation.

## How to use them (imitation learning)

Train a policy by behavior cloning: predict the action sequence from the recent observations. The
provided baseline loads these `.h5` files directly and trains a Diffusion Policy. The **main track
is state-based** (one checkpoint per level, since the state vector is level-specific); an optional
**rgb** image track is also provided.

See the challenge repository for the full training + evaluation pipeline.
