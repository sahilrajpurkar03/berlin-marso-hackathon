"""Evaluate a checkpoint on an eval config, reporting the §9.1 metrics over N episodes.
The interface here is IDENTICAL to the held-out judging harness.

  python eval.py difficulty=hard checkpoint=<path> eval_config=conf/eval/default.yaml
  # judges run the same command with the held-out config:
  python eval.py difficulty=hard checkpoint=<path> eval_config=judge/heldout.yaml

Critical behaviour:
  * Main track is STATE (obs_mode=state, the default); rgb is the optional image track.
  * Fully driven by the `eval_config` file: it supplies n_episodes, the seed list, and
    (optionally) randomisation-range OVERRIDES. Nothing about the eval conditions is
    hardcoded here.
  * A train.py checkpoint loads and runs with no code changes; default.yaml and the held-out
    config use the same pipeline -- only the randomisation values and seed list differ.
"""

import os

import hydra
import torch
from omegaconf import OmegaConf

from warehouse_sort.utils import (
    load_agent, log_run_header, make_env, print_metrics, record_eval_video, rollout_metrics,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    assert cfg.checkpoint, "pass checkpoint=<path to ckpt.pt>"
    assert cfg.get("eval_config"), "pass eval_config=<path to eval yaml>"
    log_run_header(cfg, "eval")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    eval_cfg = OmegaConf.load(cfg.eval_config)
    n_episodes = int(eval_cfg.eval.n_episodes)
    seeds = list(eval_cfg.eval.seeds)
    # randomisation: use the difficulty's training ranges unless the eval config overrides
    # them (held-out widens/recombines via this override).
    randomization = eval_cfg.get("randomization", None) or cfg.randomization

    obs_mode = cfg.obs_mode

    n_envs = min(cfg.num_envs, n_episodes)
    env, _ = make_env(cfg, obs_mode, randomization, num_envs=n_envs)
    agent, _ = load_agent(cfg.checkpoint, env, device, entrypoint=cfg.policy)

    m = rollout_metrics(env, agent, device, n_episodes, seeds, cfg.max_episode_steps)
    print_metrics("EVAL", cfg.difficulty.name, obs_mode, m,
                  hard=(cfg.difficulty.name == "hard"))
    env.close()

    # every eval run also saves a video (RecordEpisode, all views: render + scene sensor cam)
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    vid_dir = os.path.join(out_dir, "videos")
    n_vid = min(4, n_envs)
    record_eval_video(cfg, obs_mode, randomization, agent, device, vid_dir,
                      n_envs=n_vid, seed=int(seeds[0]))
    print(f"[eval] saved rollout video (render + sensor views) -> {vid_dir}", flush=True)


if __name__ == "__main__":
    main()
