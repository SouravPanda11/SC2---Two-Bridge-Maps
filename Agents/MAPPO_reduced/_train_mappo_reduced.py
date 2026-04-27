from __future__ import annotations

import atexit
import copy
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import gc
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TOTAL_TIMESTEPS = 2_000_000
DEFAULT_SAVE_INTERVAL = 50_000
DEFAULT_NUM_SEEDS = 3
DEFAULT_NUM_ENVS = 3

AGENT_NAME = "MAPPO_reduced"

MAP_ENV_MODULES = {
    "V1_Base": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V1_Base",
    "V1_Combat": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V1_Combat",
    "V1_Navigate": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V1_Navigate",
    "V2_Base": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V2_Base",
    "V2_Combat": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V2_Combat",
    "V2_Navigate": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V2_Navigate",
    "V3_Base": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V3_Base",
    "V3_Combat": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V3_Combat",
    "V3_Navigate": "Environments.MAPPO_reduced.TB_env_MAPPO_reduced_V3_Navigate",
}


@dataclass
class TrainConfig:
    map_name: str
    run_mode: str = "fresh_start"
    seed: Optional[int] = None
    seed_values: tuple[int, ...] = ()
    num_seeds: int = DEFAULT_NUM_SEEDS
    num_envs: int = DEFAULT_NUM_ENVS
    agent_name: str = AGENT_NAME

    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    save_interval: int = DEFAULT_SAVE_INTERVAL
    log_interval: int = 5_000
    eval_interval: int = 50_000
    eval_episodes: int = 5
    eval_during_training: bool = False

    rollout_steps: int = 512
    minibatch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    standardise_rewards: bool = True

    hidden_dim: int = 128
    minimap_embed_dim: int = 64
    use_minimap_actor: bool = False
    use_minimap_critic: bool = True
    obs_agent_id: bool = True
    include_player_relative: bool = True

    visualize: bool = False
    realtime: bool = False
    episode_limit: Optional[int] = None
    replay_dir: str = ""
    save_replay_episodes: int = 0
    resume_checkpoint: str = ""
    summary_window: int = 20
    use_tensorboard: bool = True
    smoke_test: bool = False

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def make_config(map_name: str, **overrides) -> TrainConfig:
    config = TrainConfig(map_name=map_name)
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise TypeError(f"Unknown MAPPO config field: {key}")
        setattr(config, key, value)
    return config


def normalize_config(config: TrainConfig) -> TrainConfig:
    config = copy.deepcopy(config)
    if config.run_mode not in {"fresh_start", "load_last_checkpoint"}:
        raise ValueError("run_mode must be 'fresh_start' or 'load_last_checkpoint'.")
    if config.map_name not in MAP_ENV_MODULES:
        known = ", ".join(sorted(MAP_ENV_MODULES))
        raise ValueError(f"Unsupported map_name={config.map_name!r}. Known: {known}")
    if config.seed is not None and config.seed_values:
        raise ValueError("Use either seed or seed_values, not both.")

    if config.smoke_test:
        config.seed = 0 if config.seed is None and not config.seed_values else config.seed
        config.num_seeds = 1
        config.num_envs = 1
        config.total_timesteps = 32
        config.save_interval = 32
        config.log_interval = 32
        config.eval_during_training = False
        config.rollout_steps = 16
        config.minibatch_size = 16
        config.update_epochs = 1

    if config.num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if config.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if config.total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    if config.save_interval < 1:
        raise ValueError("save_interval must be at least 1")
    if config.save_interval > config.total_timesteps:
        raise ValueError("save_interval cannot exceed total_timesteps")
    if config.rollout_steps < 2:
        raise ValueError("rollout_steps must be at least 2")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be at least 1")
    if config.update_epochs < 1:
        raise ValueError("update_epochs must be at least 1")
    return config


def load_env_class(map_name: str):
    module = importlib.import_module(MAP_ENV_MODULES[map_name])
    return module.TwoBridgeEnv


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_training_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def generate_random_seeds(num_seeds: int) -> tuple[int, ...]:
    rng = random.SystemRandom()
    seeds = []
    seen = set()
    while len(seeds) < num_seeds:
        seed = rng.randrange(0, 2**31 - 1)
        if seed in seen:
            continue
        seeds.append(seed)
        seen.add(seed)
    return tuple(seeds)


def resolve_seeds(config: TrainConfig) -> tuple[int, ...]:
    if config.seed_values:
        return tuple(int(seed) for seed in config.seed_values)
    if config.seed is not None:
        return (int(config.seed),)
    return generate_random_seeds(config.num_seeds)


def step_label(env_steps: int) -> str:
    if env_steps % 1_000_000 == 0:
        return f"{env_steps // 1_000_000}M"
    if env_steps % 1_000 == 0:
        return f"{env_steps // 1_000}K"
    return str(env_steps)


def next_multiple(current: int, interval: int) -> int:
    if interval <= 0:
        return current
    return ((current // interval) + 1) * interval


def get_agent_save_root(config: TrainConfig) -> Path:
    return (
        PROJECT_ROOT
        / "Agents"
        / "MAPPO_reduced"
        / config.map_name
        / "saved_models"
        / config.agent_name
    )


def get_agent_tb_root(config: TrainConfig) -> Path:
    return PROJECT_ROOT / "tb_logs" / "MAPPO_reduced" / config.map_name / config.agent_name


def get_seed_output_dirs(config: TrainConfig, seed: int) -> tuple[Path, Path]:
    save_dir = get_agent_save_root(config) / f"seed_{seed}"
    tb_log_dir = get_agent_tb_root(config) / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, tb_log_dir


def write_seed_manifest(config: TrainConfig, seeds: tuple[int, ...]) -> Path:
    manifest = asdict(config)
    manifest.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seeds": list(seeds),
            "save_root": get_agent_save_root(config).as_posix(),
            "tensorboard_root": get_agent_tb_root(config).as_posix(),
        }
    )
    save_root = get_agent_save_root(config)
    save_root.mkdir(parents=True, exist_ok=True)
    latest_manifest_path = save_root / "latest_run_manifest.json"
    dated_manifest_path = save_root / f"run_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    for path in (latest_manifest_path, dated_manifest_path):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    return latest_manifest_path


def is_final_checkpoint_name(config: TrainConfig, checkpoint_name: str) -> bool:
    return checkpoint_name == f"{config.agent_name}_final.pt"


def parse_checkpoint_steps(config: TrainConfig, checkpoint_path: Path) -> Optional[int]:
    checkpoint_name = checkpoint_path.name
    if is_final_checkpoint_name(config, checkpoint_name):
        return None
    prefix = f"{config.agent_name}_"
    suffix = ".pt"
    if not checkpoint_name.startswith(prefix) or not checkpoint_name.endswith(suffix):
        return None
    label = checkpoint_name[len(prefix) : -len(suffix)]
    match = re.fullmatch(r"(\d+)([KMB]?)", label, re.IGNORECASE)
    if match is None:
        return None
    value = int(match.group(1))
    scale = match.group(2).upper()
    return value * {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[scale]


def checkpoint_sort_key(config: TrainConfig, checkpoint_path: Path):
    parsed_steps = parse_checkpoint_steps(config, checkpoint_path)
    if parsed_steps is None:
        return None
    return parsed_steps, checkpoint_path.stat().st_mtime


def collect_seed_checkpoints(config: TrainConfig, seed: int) -> list[Path]:
    save_dir = get_agent_save_root(config) / f"seed_{seed}"
    if not save_dir.is_dir():
        return []
    checkpoint_paths = []
    for entry in save_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".pt":
            continue
        if checkpoint_sort_key(config, entry) is not None:
            checkpoint_paths.append(entry)
    return checkpoint_paths


def seed_dir_has_final(config: TrainConfig, seed_dir: Path) -> bool:
    final_name = f"{config.agent_name}_final.pt"
    return (seed_dir / final_name).is_file()


def load_latest_seed_manifest(config: TrainConfig):
    manifest_path = get_agent_save_root(config) / "latest_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No seed manifest found at {manifest_path}. Run fresh_start first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    seeds = tuple(int(seed) for seed in manifest.get("seeds", ()))
    if not seeds:
        raise RuntimeError(f"Seed manifest {manifest_path} has no seeds.")
    return manifest_path, seeds


def resolve_resume_plan(config: TrainConfig):
    manifest_path, seeds = load_latest_seed_manifest(config)
    states = []
    for seed in seeds:
        save_dir = get_agent_save_root(config) / f"seed_{seed}"
        checkpoints = collect_seed_checkpoints(config, seed)
        checkpoint = max(checkpoints, key=lambda path: checkpoint_sort_key(config, path)) if checkpoints else None
        states.append(
            {
                "seed": seed,
                "save_dir": save_dir,
                "has_final": save_dir.is_dir() and seed_dir_has_final(config, save_dir),
                "checkpoint_path": checkpoint,
            }
        )
    pending = [state for state in states if not state["has_final"]]
    if not pending:
        raise FileNotFoundError(
            "No unfinished seed found. Every seed in latest_run_manifest.json has a final checkpoint."
        )
    return manifest_path, states, pending


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _make_env_payload(env, obs):
    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "state": np.asarray(env.get_state(), dtype=np.float32),
        "minimap": np.asarray(env.get_minimap(), dtype=np.uint8),
        "avail_actions": np.asarray(env.get_avail_actions(), dtype=np.float32),
    }


def mappo_env_worker(remote, parent_remote, rank: int, env_module: str, env_kwargs: dict):
    parent_remote.close()
    worker_tmp_dir = tempfile.mkdtemp(prefix=f"tbm-mappo-worker-{rank}-")
    cleanup = lambda path=worker_tmp_dir: shutil.rmtree(path, ignore_errors=True)
    atexit.register(cleanup)

    try:
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = worker_tmp_dir
        time.sleep(0.5 * rank)

        module = importlib.import_module(env_module)
        env = module.TwoBridgeEnv(**env_kwargs)
        while True:
            cmd, data = remote.recv()
            if cmd == "get_env_info":
                remote.send(env.get_env_info())
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(_make_env_payload(env, obs))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                payload = _make_env_payload(env, obs)
                payload.update(
                    {
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": info,
                    }
                )
                remote.send(payload)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd!r}")
    except EOFError:
        pass
    except Exception:
        try:
            remote.send({"__worker_error__": traceback.format_exc()})
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass
        cleanup()


class ParallelEnvBatch:
    def __init__(self, num_envs: int, base_seed: int, map_name: str, env_kwargs: dict):
        self.num_envs = int(num_envs)
        self.closed = False
        self.ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        env_module = MAP_ENV_MODULES[map_name]

        for rank in range(self.num_envs):
            parent_remote, worker_remote = self.ctx.Pipe()
            worker_kwargs = dict(env_kwargs)
            worker_kwargs["seed"] = int(base_seed + rank)
            process = self.ctx.Process(
                target=mappo_env_worker,
                args=(worker_remote, parent_remote, rank, env_module, worker_kwargs),
            )
            process.daemon = True
            process.start()
            worker_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

        self.env_info = self.call(0, "get_env_info")

    def call(self, index: int, cmd: str, data=None):
        remote = self.remotes[index]
        remote.send((cmd, data))
        payload = remote.recv()
        self._raise_if_worker_error(payload)
        return payload

    def call_many(self, indices, cmd: str, payloads=None):
        if payloads is None:
            payloads = [None] * len(indices)
        for index, payload in zip(indices, payloads):
            self.remotes[index].send((cmd, payload))
        results = []
        for index in indices:
            payload = self.remotes[index].recv()
            self._raise_if_worker_error(payload)
            results.append((index, payload))
        return results

    def reset(self, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return self.call_many(indices, "reset")

    def get_env_info(self):
        return dict(self.env_info)

    def step(self, actions_batch, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        payloads = [np.asarray(actions_batch[index], dtype=np.int64) for index in indices]
        return self.call_many(indices, "step", payloads)

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass
        self.closed = True

    @staticmethod
    def _raise_if_worker_error(payload):
        if isinstance(payload, dict) and "__worker_error__" in payload:
            raise RuntimeError(payload["__worker_error__"])


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, values: torch.Tensor):
        if values.numel() == 0:
            return
        values = values.reshape(-1, values.size(-1))
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        self.update_from_moments(batch_mean, batch_var, values.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


class MinimapEncoder(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flat_dim = self.conv(dummy).reshape(1, -1).size(1)
        self.fc = nn.Linear(flat_dim, embed_dim)

    def forward(self, minimap: torch.Tensor):
        x = minimap.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        return self.net(inputs)


class MAPPOPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        minimap_shape: tuple[int, int, int],
        n_agents: int,
        n_actions: int,
        config: TrainConfig,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.config = config
        self.use_any_minimap = config.use_minimap_actor or config.use_minimap_critic
        self.minimap_encoder = None
        if self.use_any_minimap:
            c, h, w = minimap_shape
            self.minimap_encoder = MinimapEncoder(c, h, w, config.minimap_embed_dim)

        actor_dim = obs_dim
        if config.use_minimap_actor:
            actor_dim += config.minimap_embed_dim
        if config.obs_agent_id:
            actor_dim += n_agents
        critic_dim = state_dim
        if config.use_minimap_critic:
            critic_dim += config.minimap_embed_dim
        critic_dim += n_agents

        self.actor = MLP(actor_dim, config.hidden_dim, n_actions)
        self.critic = MLP(critic_dim, config.hidden_dim, 1)

    def encode_minimap(self, minimap: torch.Tensor):
        if self.minimap_encoder is None:
            return None
        return self.minimap_encoder(minimap)

    def _agent_ids(self, batch_size: int, device):
        return torch.eye(self.n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    def actor_logits(self, obs: torch.Tensor, minimap_embed: Optional[torch.Tensor]):
        batch_size = obs.size(0)
        pieces = [obs]
        if self.config.use_minimap_actor:
            pieces.append(minimap_embed.unsqueeze(1).expand(-1, self.n_agents, -1))
        if self.config.obs_agent_id:
            pieces.append(self._agent_ids(batch_size, obs.device))
        inputs = torch.cat(pieces, dim=-1).reshape(batch_size * self.n_agents, -1)
        logits = self.actor(inputs)
        return logits.view(batch_size, self.n_agents, self.n_actions)

    def values(self, state: torch.Tensor, minimap_embed: Optional[torch.Tensor]):
        batch_size = state.size(0)
        state_pieces = [state]
        if self.config.use_minimap_critic:
            state_pieces.append(minimap_embed)
        shared = torch.cat(state_pieces, dim=-1).unsqueeze(1).expand(-1, self.n_agents, -1)
        inputs = torch.cat([shared, self._agent_ids(batch_size, state.device)], dim=-1)
        return self.critic(inputs.reshape(batch_size * self.n_agents, -1)).view(batch_size, self.n_agents)

    def masked_distribution(self, logits: torch.Tensor, avail_actions: torch.Tensor):
        avail_bool = avail_actions.bool()
        safe_avail = avail_bool.clone()
        empty = safe_avail.sum(dim=-1) == 0
        if empty.any():
            safe_avail[empty, 0] = True
        masked_logits = logits.masked_fill(~safe_avail, -1e10)
        return Categorical(logits=masked_logits)

    @torch.no_grad()
    def act(self, obs, state, minimap, avail_actions, greedy: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=obs_t.device)
        minimap_t = torch.as_tensor(minimap, dtype=torch.uint8, device=obs_t.device)
        avail_t = torch.as_tensor(avail_actions, dtype=torch.bool, device=obs_t.device)
        minimap_embed = self.encode_minimap(minimap_t)
        logits = self.actor_logits(obs_t, minimap_embed)
        dist = self.masked_distribution(logits, avail_t)
        if greedy:
            actions = logits.masked_fill(~avail_t, -1e10).argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.values(state_t, minimap_embed)
        return (
            actions.cpu().numpy().astype(np.int64),
            log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    def evaluate_actions(self, obs, state, minimap, avail_actions, actions):
        minimap_embed = self.encode_minimap(minimap)
        logits = self.actor_logits(obs, minimap_embed)
        dist = self.masked_distribution(logits, avail_actions)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.values(state, minimap_embed)
        return log_probs, entropy, values


class MAPPOTrainer:
    def __init__(self, config: TrainConfig, seed: int):
        self.config = config
        self.seed = int(seed)
        self.device = config.device
        self.save_dir, self.tb_log_dir = get_seed_output_dirs(config, self.seed)
        self.writer = None
        if config.use_tensorboard:
            if SummaryWriter is None:
                print("TensorBoard unavailable. Disabling tensorboard logging.")
            else:
                self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))

        env_kwargs = {
            "map_name": config.map_name,
            "episode_limit": config.episode_limit,
            "visualize": config.visualize,
            "realtime": config.realtime,
            "replay_dir": config.replay_dir,
            "save_replay_episodes": config.save_replay_episodes,
            "include_player_relative": config.include_player_relative,
        }
        self.train_env = ParallelEnvBatch(
            num_envs=config.num_envs,
            base_seed=self.seed,
            map_name=config.map_name,
            env_kwargs=env_kwargs,
        )
        self.eval_env = None
        if config.eval_during_training:
            env_cls = load_env_class(config.map_name)
            self.eval_env = env_cls(seed=self.seed + 10_000, **env_kwargs)

        env_info = self.train_env.get_env_info()
        self.n_agents = int(env_info["n_agents"])
        self.n_actions = int(env_info["n_actions"])
        self.obs_dim = int(env_info["obs_shape"])
        self.state_dim = int(env_info["state_shape"])
        self.minimap_shape = tuple(int(dim) for dim in env_info["minimap_shape"])

        self.policy = MAPPOPolicy(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            minimap_shape=self.minimap_shape,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            config=config,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.reward_ms = RunningMeanStd(shape=(1,), device=self.device) if config.standardise_rewards else None

        self.env_steps = 0
        self.episode_count = 0
        self.update_count = 0
        self.recent_returns = deque(maxlen=config.summary_window)
        self.recent_lengths = deque(maxlen=config.summary_window)
        self.recent_outcomes = deque(maxlen=config.summary_window)
        self.next_save_step = next_multiple(self.env_steps, config.save_interval)
        self.next_log_step = next_multiple(self.env_steps, config.log_interval)
        self.next_eval_step = next_multiple(self.env_steps, config.eval_interval)
        self.current = None
        self.episode_returns = np.zeros(config.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(config.num_envs, dtype=np.int32)
        self._maybe_load_checkpoint()
        self._write_run_config()

    def train(self):
        self._reset_training_envs()
        print(
            "Starting MAPPO training | "
            f"seed={self.seed} | map={self.config.map_name} | "
            f"num_envs={self.config.num_envs} | rollout_steps={self.config.rollout_steps} | "
            f"device={self.device}"
        )
        while self.env_steps < self.config.total_timesteps:
            rollout = self.collect_rollout()
            stats = self.update(rollout)
            self.log_train_stats(stats)

            if self.env_steps >= self.next_log_step:
                self.print_progress()
                self.next_log_step = next_multiple(self.env_steps, self.config.log_interval)
            if self.config.eval_during_training and self.env_steps >= self.next_eval_step:
                self.log_eval(self.evaluate_policy())
                self.next_eval_step = next_multiple(self.env_steps, self.config.eval_interval)
            if self.env_steps >= self.next_save_step:
                self.save_checkpoint(step_label(self.env_steps))
                self.next_save_step = next_multiple(self.env_steps, self.config.save_interval)

        self.save_checkpoint("final")

    def _reset_training_envs(self):
        reset_results = self.train_env.reset()
        self.current = self._empty_current()
        for index, payload in reset_results:
            self._set_current(index, payload)

    def _empty_current(self):
        return {
            "obs": np.zeros((self.config.num_envs, self.n_agents, self.obs_dim), dtype=np.float32),
            "state": np.zeros((self.config.num_envs, self.state_dim), dtype=np.float32),
            "minimap": np.zeros((self.config.num_envs, *self.minimap_shape), dtype=np.uint8),
            "avail_actions": np.zeros((self.config.num_envs, self.n_agents, self.n_actions), dtype=np.float32),
        }

    def _set_current(self, index: int, payload: dict):
        self.current["obs"][index] = payload["obs"]
        self.current["state"][index] = payload["state"]
        self.current["minimap"][index] = payload["minimap"]
        self.current["avail_actions"][index] = payload["avail_actions"]

    def collect_rollout(self):
        steps = self.config.rollout_steps
        num_envs = self.config.num_envs
        rollout = {
            "obs": np.zeros((steps, num_envs, self.n_agents, self.obs_dim), dtype=np.float32),
            "state": np.zeros((steps, num_envs, self.state_dim), dtype=np.float32),
            "minimap": np.zeros((steps, num_envs, *self.minimap_shape), dtype=np.uint8),
            "avail_actions": np.zeros((steps, num_envs, self.n_agents, self.n_actions), dtype=np.float32),
            "actions": np.zeros((steps, num_envs, self.n_agents), dtype=np.int64),
            "log_probs": np.zeros((steps, num_envs, self.n_agents), dtype=np.float32),
            "values": np.zeros((steps, num_envs, self.n_agents), dtype=np.float32),
            "rewards": np.zeros((steps, num_envs), dtype=np.float32),
            "dones": np.zeros((steps, num_envs), dtype=np.float32),
        }

        for step in range(steps):
            rollout["obs"][step] = self.current["obs"]
            rollout["state"][step] = self.current["state"]
            rollout["minimap"][step] = self.current["minimap"]
            rollout["avail_actions"][step] = self.current["avail_actions"]

            actions, log_probs, values = self.policy.act(
                self.current["obs"],
                self.current["state"],
                self.current["minimap"],
                self.current["avail_actions"],
                greedy=False,
            )
            rollout["actions"][step] = actions
            rollout["log_probs"][step] = log_probs
            rollout["values"][step] = values

            step_results = self.train_env.step(actions)
            reset_indices = []
            for index, payload in step_results:
                reward = float(payload["reward"])
                done = bool(payload["terminated"] or payload["truncated"])
                rollout["rewards"][step, index] = reward
                rollout["dones"][step, index] = float(done)
                self.episode_returns[index] += reward
                self.episode_lengths[index] += 1
                self.env_steps += 1
                self._set_current(index, payload)
                if done:
                    info = payload.get("info", {})
                    self._record_episode(index, info)
                    reset_indices.append(index)

            if reset_indices:
                for index, payload in self.train_env.reset(reset_indices):
                    self._set_current(index, payload)

        with torch.no_grad():
            _, _, last_values = self.policy.act(
                self.current["obs"],
                self.current["state"],
                self.current["minimap"],
                self.current["avail_actions"],
                greedy=True,
            )
        rollout["last_values"] = last_values.astype(np.float32)
        return rollout

    def _record_episode(self, index: int, info: dict):
        result = info.get("result") or "episode_end"
        self.recent_returns.append(float(self.episode_returns[index]))
        self.recent_lengths.append(int(self.episode_lengths[index]))
        self.recent_outcomes.append(result)
        self.episode_count += 1
        if self.writer is not None:
            self.writer.add_scalar("train/episode_return", float(self.episode_returns[index]), self.env_steps)
            self.writer.add_scalar("train/episode_length", int(self.episode_lengths[index]), self.env_steps)
            for key, value in info.get("rew", {}).items():
                self.writer.add_scalar(f"train_rew/{key}", float(value), self.env_steps)
        self.episode_returns[index] = 0.0
        self.episode_lengths[index] = 0

    def compute_advantages(self, rollout):
        rewards = torch.as_tensor(rollout["rewards"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.as_tensor(rollout["dones"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        values = torch.as_tensor(rollout["values"], dtype=torch.float32, device=self.device)
        last_values = torch.as_tensor(rollout["last_values"], dtype=torch.float32, device=self.device)

        if self.reward_ms is not None:
            self.reward_ms.update(rewards)
            rewards = (rewards - self.reward_ms.mean) / torch.sqrt(self.reward_ms.var + 1e-8)

        rewards = rewards.expand(-1, -1, self.n_agents)
        dones = dones.expand(-1, -1, self.n_agents)
        advantages = torch.zeros_like(values)
        last_gae = torch.zeros((self.config.num_envs, self.n_agents), dtype=torch.float32, device=self.device)
        for step in reversed(range(self.config.rollout_steps)):
            next_values = last_values if step == self.config.rollout_steps - 1 else values[step + 1]
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_values * next_nonterminal - values[step]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std
        return advantages, returns

    def update(self, rollout):
        advantages, returns = self.compute_advantages(rollout)
        steps, num_envs = self.config.rollout_steps, self.config.num_envs
        transition_count = steps * num_envs

        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device).view(transition_count, self.n_agents, self.obs_dim)
        state = torch.as_tensor(rollout["state"], dtype=torch.float32, device=self.device).view(transition_count, self.state_dim)
        minimap = torch.as_tensor(rollout["minimap"], dtype=torch.uint8, device=self.device).view(transition_count, *self.minimap_shape)
        avail = torch.as_tensor(rollout["avail_actions"], dtype=torch.bool, device=self.device).view(transition_count, self.n_agents, self.n_actions)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device).view(transition_count, self.n_agents)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device).view(transition_count, self.n_agents)
        advantages = advantages.view(transition_count, self.n_agents)
        returns = returns.view(transition_count, self.n_agents)

        stats = Counter()
        updates = 0
        indices = np.arange(transition_count)
        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, transition_count, self.config.minibatch_size):
                mb = torch.as_tensor(indices[start : start + self.config.minibatch_size], dtype=torch.long, device=self.device)
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    obs[mb], state[mb], minimap[mb], avail[mb], actions[mb]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                pg_loss_1 = -advantages[mb] * ratio
                pg_loss_2 = -advantages[mb] * torch.clamp(
                    ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
                )
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                value_loss = 0.5 * torch.square(returns[mb] - values).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[mb] - new_log_probs).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                stats["grad_norm"] += float(grad_norm.item())
                updates += 1

        self.update_count += 1
        return {key: value / max(updates, 1) for key, value in stats.items()}

    def evaluate_policy(self):
        if self.eval_env is None:
            return {}
        returns = []
        lengths = []
        outcomes = Counter()
        for _ in range(self.config.eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_return = 0.0
            ep_len = 0
            while True:
                payload = _make_env_payload(self.eval_env, obs)
                actions, _, _ = self.policy.act(
                    np.expand_dims(payload["obs"], 0),
                    np.expand_dims(payload["state"], 0),
                    np.expand_dims(payload["minimap"], 0),
                    np.expand_dims(payload["avail_actions"], 0),
                    greedy=True,
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(actions[0])
                ep_return += float(reward)
                ep_len += 1
                if terminated or truncated:
                    outcomes[info.get("result") or "episode_end"] += 1
                    break
            returns.append(ep_return)
            lengths.append(ep_len)
        return {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "length_mean": float(np.mean(lengths)),
            "outcomes": dict(outcomes),
        }

    def log_train_stats(self, stats: dict):
        if self.writer is None:
            return
        for key, value in stats.items():
            self.writer.add_scalar(f"loss/{key}", value, self.env_steps)

    def log_eval(self, stats: dict):
        if not stats:
            return
        print(
            "Eval | "
            f"env_steps={self.env_steps} | return_mean={stats['return_mean']:.3f} | "
            f"length_mean={stats['length_mean']:.1f} | outcomes={stats['outcomes']}"
        )
        if self.writer is None:
            return
        self.writer.add_scalar("eval/return_mean", stats["return_mean"], self.env_steps)
        self.writer.add_scalar("eval/return_std", stats["return_std"], self.env_steps)
        self.writer.add_scalar("eval/length_mean", stats["length_mean"], self.env_steps)

    def print_progress(self):
        mean_return = float(np.mean(self.recent_returns)) if self.recent_returns else 0.0
        mean_length = float(np.mean(self.recent_lengths)) if self.recent_lengths else 0.0
        print(
            "Train | "
            f"env_steps={self.env_steps} | episodes={self.episode_count} | "
            f"updates={self.update_count} | mean_return={mean_return:.3f} | "
            f"mean_length={mean_length:.1f} | recent_outcomes={dict(Counter(self.recent_outcomes))}"
        )

    def save_checkpoint(self, label: str):
        checkpoint = {
            "config": asdict(self.config),
            "seed": self.seed,
            "env_steps": self.env_steps,
            "episode_count": self.episode_count,
            "update_count": self.update_count,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "recent_returns": list(self.recent_returns),
            "recent_lengths": list(self.recent_lengths),
            "recent_outcomes": list(self.recent_outcomes),
        }
        if self.reward_ms is not None:
            checkpoint["reward_ms"] = {
                "mean": self.reward_ms.mean.detach().cpu(),
                "var": self.reward_ms.var.detach().cpu(),
                "count": self.reward_ms.count,
            }
        checkpoint_path = self.save_dir / f"{self.config.agent_name}_{label}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        release_training_memory()

    def _maybe_load_checkpoint(self):
        if not self.config.resume_checkpoint:
            return
        checkpoint_path = Path(self.config.resume_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.episode_count = int(checkpoint.get("episode_count", 0))
        self.update_count = int(checkpoint.get("update_count", 0))
        self.recent_returns = deque(checkpoint.get("recent_returns", []), maxlen=self.config.summary_window)
        self.recent_lengths = deque(checkpoint.get("recent_lengths", []), maxlen=self.config.summary_window)
        self.recent_outcomes = deque(checkpoint.get("recent_outcomes", []), maxlen=self.config.summary_window)
        reward_ms = checkpoint.get("reward_ms")
        if reward_ms is not None and self.reward_ms is not None:
            self.reward_ms.mean.copy_(reward_ms["mean"].to(self.device))
            self.reward_ms.var.copy_(reward_ms["var"].to(self.device))
            self.reward_ms.count = reward_ms["count"]
        self.next_save_step = next_multiple(self.env_steps, self.config.save_interval)
        self.next_log_step = next_multiple(self.env_steps, self.config.log_interval)
        print(f"Resumed from checkpoint: {checkpoint_path}")

    def _write_run_config(self):
        config_path = self.save_dir / "run_config.json"
        payload = asdict(self.config)
        payload["seed"] = self.seed
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.eval_env is not None:
            self.eval_env.close()
        if self.train_env is not None:
            self.train_env.close()


def train_for_seed(config: TrainConfig, seed: int):
    set_global_seeds(seed)
    start = time.perf_counter()
    trainer = MAPPOTrainer(copy.deepcopy(config), seed)
    try:
        trainer.train()
        wall = time.perf_counter() - start
        return {
            "seed": seed,
            "env_steps": trainer.env_steps,
            "episodes": trainer.episode_count,
            "updates": trainer.update_count,
            "seed_wall": wall,
        }
    finally:
        trainer.close()
        del trainer
        release_training_memory()


def train(config: TrainConfig):
    overall_start = time.perf_counter()
    config = normalize_config(config)
    resume_plan = None
    if config.run_mode == "fresh_start":
        seeds = resolve_seeds(config)
        manifest_path = write_seed_manifest(config, seeds)
    else:
        manifest_path, seed_states, pending_states = resolve_resume_plan(config)
        seeds = tuple(state["seed"] for state in pending_states)
        resume_plan = {state["seed"]: state["checkpoint_path"] for state in pending_states}
        completed = tuple(state["seed"] for state in seed_states if state["has_final"])
        if completed:
            print(f"Resume skip | completed_seeds={completed}")

    print(f"Seed manifest: {manifest_path}")
    print(
        "Run plan | "
        f"mode={config.run_mode} | map={config.map_name} | seeds={seeds} | "
        f"num_envs={config.num_envs} | total_timesteps={config.total_timesteps} | "
        f"save_interval={config.save_interval}"
    )

    results = []
    for seed in seeds:
        seed_config = copy.deepcopy(config)
        if resume_plan is not None:
            checkpoint = resume_plan.get(seed)
            seed_config.resume_checkpoint = str(checkpoint) if checkpoint else ""
        results.append(train_for_seed(seed_config, seed))

    overall_wall = time.perf_counter() - overall_start
    print("Per-seed recap")
    for result in results:
        print(
            "  - "
            f"seed={result['seed']} | env_steps={result['env_steps']} | "
            f"episodes={result['episodes']} | updates={result['updates']} | "
            f"seed_wall={result['seed_wall']:.2f}s"
        )
    print(
        "Overall MAPPO run summary | "
        f"map={config.map_name} | num_seeds={len(results)} | overall_wall={overall_wall:.2f}s"
    )
    return results


def train_with_settings(map_name: str, **overrides):
    if "append_minimap_to_state" in overrides:
        overrides["use_minimap_critic"] = overrides.pop("append_minimap_to_state")
    if "append_minimap_to_obs" in overrides:
        overrides["use_minimap_actor"] = overrides.pop("append_minimap_to_obs")
    return train(make_config(map_name, **overrides))
