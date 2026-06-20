
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import warehouse_sort  # noqa: F401  (registers WarehouseSort-v1)


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "WarehouseSort-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 200
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 5000
    """evaluation frequency in terms of raw environment steps (NOT iterations, despite the name)"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_ee_delta_pos"
    """the control mode to use for the environment"""
    sim_backend: str = "gpu"
    """the simulation backend to use (passed by il/train.py's dispatcher as --sim-backend)"""
    max_episode_steps: int = 200
    """env max episode steps (shared with the rest of the repo's config)"""

    # Algorithm specific arguments
    total_timesteps: int = 50_000
    """total timesteps of the experiments"""
    buffer_size: int = 200_000
    """the ONLINE replay memory buffer size (separate from demo_buffer_size -- see seed_replay_buffer_from_demos)"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.95
    """the discount factor gamma. NOTE: the upstream default (0.8) gives an effective horizon of
    only ~20-30 steps (0.8^115≈7.6e-12), far too short for our ~115-400 step episodes -- this was
    diagnosed as the likely root cause of Q-value collapse in the RGB SAC baseline. 0.99 was tried
    too but caused runaway Q-value divergence; 0.95 is the validated middle ground."""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 512
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # WarehouseSort-specific: demo-seeded replay buffer (RLPD-style, see Ball et al. 2023)
    demo_path: Optional[str] = None
    """path to a WarehouseSort demo .h5 (any obs mode -- only the recorded ACTIONS are used; a
    fresh env instance is replayed with obs_mode=state regardless of how the demo was recorded,
    so existing rgb-recorded demos work fine here without needing a separate state-format dump)."""
    num_seed_demos: Optional[int] = None
    """cap on how many demo episodes to replay for seeding (None = use all in the demo file)"""
    demo_buffer_size: int = 30_000
    """capacity of the SEPARATE demo replay buffer (never overwritten by online data). State
    vectors are tiny vs rgb frames, so this can comfortably hold every transition with room to
    spare -- no GPU memory pressure like the image version had."""
    demo_ratio: float = 0.5
    """RLPD-style 50/50 batch composition: fraction of each training batch sampled from the
    demo buffer (rest from the online buffer), kept constant for the whole run."""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

    def add(self, obs, next_obs, action, reward, done):
        if self.storage_device == torch.device("cpu"):
            obs, next_obs, action, reward, done = obs.cpu(), next_obs.cpu(), action.cpu(), reward.cpu(), done.cpu()
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, max(self.pos, 1), size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


def make_mlp(in_dim, hidden, out_dim, layer_norm=False):
    layers = []
    c_in = in_dim
    for h in hidden:
        layers.append(nn.Linear(c_in, h))
        if layer_norm:
            # RLPD: LayerNorm in the critic MLP counters Q-value overestimation/extrapolation
            # error from off-policy bootstrapping on a mix of offline (demo) and online data.
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        c_in = h
    layers.append(nn.Linear(c_in, out_dim))
    return nn.Sequential(*layers)


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.net = make_mlp(obs_dim + act_dim, [256, 256, 256], 1, layer_norm=True)

    def forward(self, x, a):
        return self.net(torch.cat([x, a], 1))


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


def _cache_path(demo_path: str, num_seed_demos: Optional[int]):
    n = "all" if num_seed_demos is None else str(num_seed_demos)
    return demo_path[:-len(".h5")] + f".sac_state_seed_cache_{n}.pt"


def seed_replay_buffer_from_demos(rb: ReplayBuffer, demo_path: str, env_id: str, env_kwargs: dict,
                                   device, num_seed_demos: Optional[int] = None, use_cache: bool = True):
    """Replay recorded WarehouseSort demo actions through a real (single-env) STATE-obs-mode
    instance to get ground-truth (obs, next_obs, action, reward, done) transitions. Works off
    ANY existing demo .h5 (rgb or state) since only the recorded actions are used -- the fresh
    env captures real state observations (parcel_pose, parcel_tag, bin_position, bin_color,
    tcp_pose, is_grasped) regardless of what obs mode the demo file itself was recorded in.
    """
    cache_path = _cache_path(demo_path, num_seed_demos)
    if use_cache and os.path.exists(cache_path):
        print(f"[seed] loading cached demo transitions from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        flat_idx = 0
        capacity = rb.per_env_buffer_size * rb.num_envs
        n = min(len(cached["actions"]), capacity)
        for i in range(n):
            pos, env_idx = i // rb.num_envs, i % rb.num_envs
            rb.obs[pos, env_idx] = cached["obs"][i].to(rb.storage_device)
            rb.next_obs[pos, env_idx] = cached["next_obs"][i].to(rb.storage_device)
            rb.actions[pos, env_idx] = cached["actions"][i].to(rb.storage_device)
            rb.rewards[pos, env_idx] = cached["rewards"][i].to(rb.storage_device)
            rb.dones[pos, env_idx] = cached["dones"][i].to(rb.storage_device)
            flat_idx = i + 1
        rb.pos = (flat_idx + rb.num_envs - 1) // rb.num_envs
        if rb.pos >= rb.per_env_buffer_size:
            rb.pos = 0
            rb.full = True
        print(f"[seed] loaded {flat_idx} cached demo transitions into replay buffer "
              f"({flat_idx / capacity * 100:.1f}% of capacity), "
              f"total reward across demos: {cached['total_reward']:.1f} "
              f"(avg {cached['total_reward'] / max(cached['n_demos'], 1):.2f}/episode)")
        return flat_idx

    json_path = demo_path[:-len(".h5")] + ".json"
    with open(json_path) as f:
        meta = json.load(f)
    h5 = h5py.File(demo_path, "r")

    state_env_kwargs = dict(env_kwargs)
    state_env_kwargs["obs_mode"] = "state"
    seed_env = gym.make(env_id, num_envs=1, reward_mode="sparse", **state_env_kwargs)
    if isinstance(seed_env.action_space, gym.spaces.Dict):
        seed_env = FlattenActionSpaceWrapper(seed_env)
    seed_env = ManiSkillVectorEnv(seed_env, 1, ignore_terminations=True, record_metrics=False)

    episodes = meta["episodes"]
    n_demos = len(episodes) if num_seed_demos is None else min(num_seed_demos, len(episodes))
    capacity = rb.per_env_buffer_size * rb.num_envs
    obs_cache, next_obs_cache, action_cache, reward_cache, done_cache = [], [], [], [], []
    flat_idx = 0
    total_reward = 0.0
    for i in tqdm.tqdm(range(n_demos), desc="seeding replay buffer from demos (state)"):
        if flat_idx >= capacity:
            print(f"[seed] replay buffer full after {i} demos, stopping early")
            break
        ep = episodes[i]
        traj_key = f"traj_{ep['episode_id']}"
        actions_np = np.asarray(h5[traj_key]["actions"])
        obs, _ = seed_env.reset(seed=[int(ep["episode_seed"])])
        for t in range(actions_np.shape[0]):
            if flat_idx >= capacity:
                break
            action = torch.tensor(actions_np[t:t + 1], dtype=torch.float32, device=device)
            next_obs, reward, terminated, truncated, info = seed_env.step(action)
            done = (terminated | truncated).float()
            total_reward += reward.sum().item()

            o, no = obs[0].cpu(), next_obs[0].cpu()
            a, r, d = action[0].cpu(), reward[0].cpu(), done[0].cpu()
            obs_cache.append(o); next_obs_cache.append(no)
            action_cache.append(a); reward_cache.append(r); done_cache.append(d)

            pos, env_idx = flat_idx // rb.num_envs, flat_idx % rb.num_envs
            rb.obs[pos, env_idx] = o.to(rb.storage_device)
            rb.next_obs[pos, env_idx] = no.to(rb.storage_device)
            rb.actions[pos, env_idx] = a.to(rb.storage_device)
            rb.rewards[pos, env_idx] = r.to(rb.storage_device)
            rb.dones[pos, env_idx] = d.to(rb.storage_device)

            obs = next_obs
            flat_idx += 1
    rb.pos = (flat_idx + rb.num_envs - 1) // rb.num_envs
    if rb.pos >= rb.per_env_buffer_size:
        rb.pos = 0
        rb.full = True
    seed_env.close()
    print(f"[seed] loaded {flat_idx} demo transitions from {n_demos} episodes "
          f"into replay buffer ({flat_idx / capacity * 100:.1f}% of capacity), "
          f"total reward across demos: {total_reward:.1f} (avg {total_reward / max(n_demos,1):.2f}/episode)")

    if use_cache:
        torch.save({
            "obs": obs_cache, "next_obs": next_obs_cache, "actions": action_cache,
            "rewards": reward_cache, "dones": done_cache,
            "total_reward": total_reward, "n_demos": n_demos,
        }, cache_path)
        print(f"[seed] cached {flat_idx} transitions to {cache_path} for instant reload next run")
    return flat_idx


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend=args.sim_backend)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    # Match the eval/train env to the demo distribution: pull the WarehouseSort scene kwargs
    # straight from the demo's recorded env_kwargs (num parcels, fixed_poses, randomization).
    if args.demo_path is not None and args.env_id.startswith("WarehouseSort"):
        with open(args.demo_path[:-len(".h5")] + ".json") as f:
            demo_info = json.load(f)
        _dk = demo_info["env_info"]["env_kwargs"]
        for _k in ("num_parcels", "fixed_poses", "randomization"):
            if _k in _dk:
                env_kwargs[_k] = _dk[_k]
    env_kwargs["max_episode_steps"] = args.max_episode_steps

    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    online_rb = ReplayBuffer(env=envs, num_envs=args.num_envs, buffer_size=args.buffer_size,
                             storage_device=torch.device(args.buffer_device), sample_device=device)
    demo_rb = ReplayBuffer(env=envs, num_envs=args.num_envs, buffer_size=args.demo_buffer_size,
                           storage_device=torch.device(args.buffer_device), sample_device=device)

    if args.demo_path is not None and not args.evaluate:
        seed_replay_buffer_from_demos(demo_rb, args.demo_path, args.env_id, env_kwargs, device,
                                       num_seed_demos=args.num_seed_demos)

    def sample_batch(batch_size):
        online_has_data = online_rb.full or online_rb.pos > 0
        if not online_has_data:
            return demo_rb.sample(batch_size)
        n_demo = int(round(batch_size * args.demo_ratio))
        n_online = batch_size - n_demo
        d, o = demo_rb.sample(n_demo), online_rb.sample(n_online)
        return ReplayBufferSample(
            obs=torch.cat([d.obs, o.obs], dim=0),
            next_obs=torch.cat([d.next_obs, o.next_obs], dim=0),
            actions=torch.cat([d.actions, o.actions], dim=0),
            rewards=torch.cat([d.rewards, o.rewards], dim=0),
            dones=torch.cat([d.dones, o.dones], dim=0),
        )

    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = demo_rb.pos > 0 or demo_rb.full

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            # WarehouseSort's custom evaluate() keys (sort_accuracy, mis_sort_count, ...) aren't
            # standard success/fail keys, so ManiSkillVectorEnv's auto "episode" metric tracking
            # doesn't forward them -- call evaluate() directly (same fix applied to the rgb SAC
            # baseline and to ACT's evaluate.py).
            if hasattr(eval_envs.unwrapped, "evaluate"):
                ev = eval_envs.unwrapped.evaluate()
                for k in ("sort_accuracy", "success_count", "mis_sort_count"):
                    if k in ev:
                        mean = ev[k].float().mean()
                        eval_metrics_mean[k] = mean
                        if logger is not None:
                            logger.add_scalar(f"eval/{k}", mean, global_step)
            desc = f"global_step={global_step}"
            if "sort_accuracy" in eval_metrics_mean:
                desc += f", sort_accuracy: {eval_metrics_mean['sort_accuracy']:.2f}"
            if "success_once" in eval_metrics_mean:
                desc += f", success_once: {eval_metrics_mean['success_once']:.2f}"
            pbar.set_description(desc)
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")

        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            if not learning_has_started:
                actions = 2 * torch.rand(size=envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
                else:
                    need_final_obs = truncations & (~terminations)
                    stop_bootstrap = terminations
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            online_rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        if global_step < args.learning_starts and not learning_has_started:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = sample_batch(args.batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_update % args.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
