"""ACT (Action Chunking Transformer) policy entrypoint for eval.py / the judge.

Satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire it in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.act_policy:load_act \\
        checkpoint=<path> eval_config=conf/eval/default.yaml

Note: uses simple chunk-replay (predict num_queries actions, play them back open-loop, then
re-predict) rather than ACT's optional temporal-ensembling eval mode -- the latter needs an
explicit per-episode reset signal that eval.py's rollout loop doesn't provide, so chunk-replay
is the lower-risk choice here (same simplification precedent as the DP policy's obs-history
wrapper, which also doesn't get an explicit reset signal between eval episodes).
"""

import torch
import torchvision.transforms as T


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


class _ACTPolicy:
    def __init__(self, agent, num_queries, act_horizon, device):
        self.agent = agent.to(device).eval()
        self.num_queries = num_queries
        self.act_horizon = act_horizon  # how many predicted actions to replay before re-querying
        self.device = device
        self.resize = T.Resize((224, 224), antialias=True)
        self._chunk = None
        self._step = 0

    def _prep_obs(self, obs):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)  # (N, H, W, 3) uint8
        rgb = rgb.permute(0, 3, 1, 2)  # (N, 3, H, W)
        rgb = self.resize(rgb)  # (N, 3, 224, 224)
        rgb = rgb.unsqueeze(1)  # (N, 1, 3, 224, 224) -- num_cams=1
        return {"state": state, "rgb": rgb}

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        if self._chunk is None or self._step >= self.act_horizon:
            obs_in = self._prep_obs(obs)
            self._chunk = self.agent.get_action(obs_in)  # (N, num_queries, act_dim), delta control -> unnormalized
            self._step = 0
        action = self._chunk[:, self._step]
        self._step += 1
        return action.clamp(-1.0, 1.0)


def load_act(checkpoint, sample_obs, action_space, device,
             backbone="resnet18", num_queries=30, act_horizon=10,
             hidden_dim=256, enc_layers=2, dec_layers=4, dim_feedforward=512,
             nheads=8, dropout=0.1, pre_norm=False, position_embedding="sine",
             masks=False, dilation=False, lr_backbone=1e-5, kl_weight=10):
    """Load an ACT checkpoint trained by il/baselines/act/train_rgbd.py (uses EMA weights).

    If you changed any architecture flag for training (backbone, hidden_dim, enc_layers,
    dec_layers, dim_feedforward, nheads), pass the same value here or the checkpoint won't
    load. ``num_queries`` is the exception -- eval.py/the judge call this function with no
    extra kwargs, but different training runs use different chunk sizes (e.g. a 60-step chunk
    for a longer max_episode_steps), so it's inferred directly from the checkpoint's
    ``query_embed.weight`` shape below rather than trusted from the default/CLI value.

    act_horizon defaults to 10, not num_queries: grading uses plain chunk-replay (no
    temporal_agg), so replaying the full chunk open-loop before re-observing is the least
    reactive option. Re-querying every ~10 steps trades a bit of compute for much better
    recovery from jitter/bin-swap (same reasoning as DP's act_horizon).
    """
    import types
    _add_baseline_path("act")
    from train_rgbd import Agent

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_agent", ckpt.get("agent"))
    ckpt_num_queries = state_dict["model.query_embed.weight"].shape[0]
    if ckpt_num_queries != num_queries:
        print(f"[act_policy] checkpoint was trained with num_queries={ckpt_num_queries} "
              f"(default is {num_queries}) -- using {ckpt_num_queries} to match the checkpoint",
              flush=True)
        num_queries = ckpt_num_queries
    act_horizon = min(act_horizon or num_queries, num_queries)

    state_dim = sample_obs["state"].shape[1]
    args = types.SimpleNamespace(
        backbone=backbone, num_queries=num_queries, hidden_dim=hidden_dim,
        enc_layers=enc_layers, dec_layers=dec_layers, dim_feedforward=dim_feedforward,
        nheads=nheads, dropout=dropout, pre_norm=pre_norm, position_embedding=position_embedding,
        masks=masks, dilation=dilation, lr_backbone=lr_backbone, kl_weight=kl_weight,
        include_depth=False,
    )
    stub = types.SimpleNamespace(
        single_observation_space={
            "state": types.SimpleNamespace(shape=(state_dim,)),
            "rgb": types.SimpleNamespace(shape=(1, 3, 224, 224)),  # (num_cams, C, H, W)
        },
        single_action_space=types.SimpleNamespace(shape=(action_space.shape[0],)),
    )

    agent = Agent(stub, args)
    agent.load_state_dict(state_dict)
    return _ACTPolicy(agent, num_queries, act_horizon, device)
