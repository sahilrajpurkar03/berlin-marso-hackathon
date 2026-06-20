"""RL (SAC) policy entrypoint for eval.py / the judge.

Satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire it in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.rl_policy:load_sac \\
        checkpoint=<path> eval_config=conf/eval/default.yaml
"""

import torch


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


class _SACPolicy:
    def __init__(self, actor, device):
        self.actor = actor.to(device).eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)
        obs_in = {"state": state, "rgb": rgb}
        if deterministic:
            action = self.actor.get_eval_action(obs_in)
        else:
            action, _, _, _ = self.actor.get_action(obs_in)
        return action.clamp(-1.0, 1.0)


def load_sac(checkpoint, sample_obs, action_space, device):
    """Load a SAC checkpoint trained by il/baselines/sac/sac_rgbd.py (uses the raw actor;
    SAC has no EMA weights, unlike the diffusion policy baseline)."""
    import types
    _add_baseline_path("sac")
    from sac_rgbd import Actor

    stub = types.SimpleNamespace(
        single_observation_space=None,  # unused by Actor.__init__ directly
        single_action_space=types.SimpleNamespace(
            shape=(action_space.shape[0],),
            high=action_space.high if hasattr(action_space, "high") else None,
            low=action_space.low if hasattr(action_space, "low") else None,
        ),
    )
    # Actor only reads envs.single_action_space.{shape,high,low} and
    # envs.single_observation_space['state'].shape[0] -- build a minimal stand-in.
    import numpy as np
    state_dim = sample_obs["state"].shape[1]
    stub.single_observation_space = {"state": types.SimpleNamespace(shape=(state_dim,))}
    if stub.single_action_space.high is None:
        stub.single_action_space.high = np.ones(action_space.shape[0], dtype=np.float32)
        stub.single_action_space.low = -np.ones(action_space.shape[0], dtype=np.float32)

    actor = Actor(stub, sample_obs=sample_obs)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    return _SACPolicy(actor, device)
