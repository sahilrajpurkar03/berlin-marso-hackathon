"""Example of the policy contract with a NON-RL, no-checkpoint policy.

Run it with:  pixi run python eval.py difficulty=easy checkpoint=x policy=examples.random_policy:load_policy \
                  eval_config=conf/eval/default.yaml
(any path works for `checkpoint` here since this policy ignores it.)

It shows that eval.py accepts ANY object with ``act(obs, deterministic=True)`` that consumes
ONLY the observation and returns actions in [-1, 1] -- no env access, no privileged state.
"""

import torch


class RandomPolicy:
    def __init__(self, action_space, device):
        self.action_dim = action_space.shape[0]
        self.device = device

    def act(self, obs, deterministic=True):
        # obs is the state vector (N, obs_dim) for the main track, or a dict
        # {"rgb": (N,128,128,3), "state": (N,26)} for the image track. We only need its
        # batch size; we never read privileged fields or the env.
        n = (obs["rgb"] if isinstance(obs, dict) else obs).shape[0]
        return torch.rand((n, self.action_dim), device=self.device) * 2 - 1


def load_policy(checkpoint, sample_obs, action_space, device):
    return RandomPolicy(action_space, device)
