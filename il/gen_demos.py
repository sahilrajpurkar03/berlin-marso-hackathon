"""Generate an imitation-learning demonstration dataset for WarehouseSort-v1.

This is the IL track's data step. It rolls out the **scripted waypoint policy**
(``examples/scripted_policy.py``) and records the episodes in ManiSkill's standard trajectory
format (``.h5`` + ``.json``) via the official ``RecordEpisode`` wrapper.

After recording the raw demos this script invokes ManiSkill's ``replay_trajectory`` tool to
produce the training-ready dataset(s): ``state`` (main track) and/or ``rgb`` (image track).

It also saves a demo **video (.mp4) and .gif** for the README / notebook.

NOTE: we ship pre-generated demos for every level, so most teams never need to run this. It is
provided for transparency and for teams that want to collect extra demos.

Usage:
  pixi run python il/gen_demos.py --difficulty easy   --num-episodes 200
  pixi run python il/gen_demos.py --difficulty medium --num-episodes 200
  pixi run python il/gen_demos.py --difficulty hard   --num-episodes 200
  pixi run python il/gen_demos.py --num-episodes 200 --no-replay     # just record raw demos
"""

import argparse
import glob
import os
import subprocess
import sys

import gymnasium as gym

# reuse the scripted policy (grasp/place logic) and its per-difficulty env kwargs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warehouse_sort  # noqa: F401  (registers WarehouseSort-v1)
from examples.scripted_policy import DIFFICULTY_KWARGS, scripted_episode
from mani_skill.utils.wrappers.record import RecordEpisode

CONTROL_MODE = "pd_ee_delta_pos"   # the fixed task controller (README §Action space)


def record_raw_demos(out_dir, difficulty, num_episodes, base_seed, max_steps, obs_camera="scene"):
    """Roll the scripted policy for ``num_episodes`` seeds and save raw .h5 + .json.

    ``obs_camera`` is baked into the recorded env_kwargs, so the later ``replay_trajectory`` step
    re-renders the rgb obs from that same camera ("scene" -> fixed third-person view)."""
    os.makedirs(out_dir, exist_ok=True)
    kwargs = DIFFICULTY_KWARGS[difficulty]
    n_parcels = kwargs["num_parcels"]

    env = gym.make(
        "WarehouseSort-v1",
        num_envs=1,
        obs_mode="state",
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        obs_camera=obs_camera,
        **kwargs,
    )
    # RecordEpisode saves one trajectory per episode (flushed on each reset / on close) into a
    # single .h5 + .json pair, recording env states too so replay_trajectory can re-render obs.
    env = RecordEpisode(
        env,
        output_dir=out_dir,
        trajectory_name="trajectory",
        save_trajectory=True,
        save_video=False,
        record_env_state=True,
        save_on_reset=True,
    )

    print(f"recording {num_episodes} episodes  difficulty={difficulty}", flush=True)
    n_success = 0
    for i in range(num_episodes):
        seed = base_seed + i
        history = scripted_episode(env, max_steps=max_steps, seed=seed)
        info = history[-1][-1]
        sc = info.get("success_count")
        sc_val = float(sc.item() if hasattr(sc, "item") else sc)
        success = sc_val >= n_parcels
        n_success += int(success)
        print(f"  ep {i:3d} seed={seed} sorted={sc_val:.0f}/{n_parcels} success={success}",
              flush=True)
    env.close()  # final flush

    h5 = os.path.join(out_dir, "trajectory.h5")
    print(f"\nrecorded {num_episodes} episodes  ({n_success} full-success)  -> {h5}", flush=True)
    return h5


def replay(h5_path, obs_mode, suffix):
    """ManiSkill's replay_trajectory -> training-ready dataset in `obs_mode`.

    Re-records the same actions (pd_ee_delta_pos) with freshly rendered observations on the GPU
    backend (single env: parallel replay would require recorded env-states this env can't emit)."""
    out = h5_path.replace(".h5", f".{suffix}.h5")
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "_replay.py"),
        "--traj-path", h5_path,
        "--obs-mode", obs_mode,
        "--sim-backend", "physx_cuda",
        "--save-traj",
        "--num-envs", "1",
    ]
    print(f"\n[replay -> {obs_mode}]  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[replay] done ({obs_mode}); see {os.path.dirname(h5_path)}", flush=True)
    return out


def _mp4_to_gif(mp4_path, gif_path, fps=15, stride=2, max_width=480):
    """Downsample an mp4 to a compact looping gif for the README / notebook."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    reader = imageio.get_reader(mp4_path)
    frames = []
    for i, frame in enumerate(reader):
        if i % stride:
            continue
        img = Image.fromarray(np.asarray(frame))
        if img.width > max_width:
            h = int(img.height * max_width / img.width)
            img = img.resize((max_width, h), Image.BILINEAR)
        frames.append(np.asarray(img))
    reader.close()
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"[media] gif -> {gif_path}  ({len(frames)} frames)", flush=True)


def record_demo_media(media_dir, difficulty, seed, max_steps):
    """Render one clean scripted episode to an mp4 (render + sensor views) and a gif."""
    os.makedirs(media_dir, exist_ok=True)
    kwargs = DIFFICULTY_KWARGS[difficulty]
    env = gym.make(
        "WarehouseSort-v1",
        num_envs=1,
        obs_mode="state",
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        render_mode="all",
        max_episode_steps=max_steps,
        **kwargs,
    )
    env = RecordEpisode(
        env, output_dir=media_dir, save_trajectory=False, save_video=True,
        video_fps=20, max_steps_per_video=max_steps,
    )
    scripted_episode(env, max_steps=max_steps, seed=seed)
    env.close()

    produced = sorted(glob.glob(os.path.join(media_dir, "*.mp4")), key=os.path.getmtime)
    mp4 = os.path.join(media_dir, f"{difficulty}_demo.mp4")
    if produced:
        os.replace(produced[-1], mp4)
        print(f"[media] mp4 -> {mp4}", flush=True)
        _mp4_to_gif(mp4, os.path.join(media_dir, f"{difficulty}_demo.gif"))
    else:
        print("[media] WARNING: no mp4 produced", flush=True)


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--num-episodes", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="per-episode cap; default scales with parcel count")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-replay", action="store_true",
                    help="only record raw demos; skip the replay_trajectory obs conversion")
    ap.add_argument("--obs-modes", nargs="*", default=["state"],
                    help="which obs representations to produce via replay_trajectory "
                         "(main track -> state; add 'rgb' for the optional image track)")
    ap.add_argument("--obs-camera", default="scene", choices=["scene"],
                    help="image obs camera (scene = fixed third-person; the only supported camera)")
    ap.add_argument("--no-media", action="store_true",
                    help="skip the demo mp4 + gif")
    ap.add_argument("--media-dir", default=None, help="where to write the demo mp4 + gif")
    args = ap.parse_args()

    n_parcels = DIFFICULTY_KWARGS[args.difficulty]["num_parcels"]
    max_steps = args.max_steps or max(150, 70 * n_parcels)
    out_dir = args.out_dir or os.path.join(repo, "il", "demos", args.difficulty)

    h5 = record_raw_demos(out_dir, args.difficulty, args.num_episodes, args.base_seed,
                          max_steps, obs_camera=args.obs_camera)

    if not args.no_replay:
        for om in args.obs_modes:
            replay(h5, om, om)

    if not args.no_media:
        media_dir = args.media_dir or os.path.join(repo, "media")
        record_demo_media(media_dir, args.difficulty, args.base_seed, max_steps)


if __name__ == "__main__":
    main()
