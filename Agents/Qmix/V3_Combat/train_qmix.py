import atexit
import copy
import gc
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import time
import sys
import tempfile
import traceback
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Environments.MultiAgent.TB_env_QMIX_V3_Combat import TwoBridgeEnv


DEFAULT_TOTAL_TIMESTEPS = 2_000_000
DEFAULT_SAVE_INTERVAL = 50_000
DEFAULT_NUM_SEEDS = 3
DEFAULT_NUM_ENVS = 3

AGENT_NAME = "QMIX"
MAP_NAME = "V3_Combat"

# ============================================================================
# Run mode: comment/uncomment exactly one option below.
# ============================================================================
RUN_MODE = "fresh_start"
# RUN_MODE = "load_last_checkpoint"

# Fresh start settings.
FRESH_START_SEED = None
FRESH_START_SEED_VALUES: tuple[int, ...] = ()

# Training settings.
TOTAL_TIMESTEPS = DEFAULT_TOTAL_TIMESTEPS
SAVE_INTERVAL = DEFAULT_SAVE_INTERVAL
LOG_INTERVAL = 5_000
NUM_SEEDS = DEFAULT_NUM_SEEDS
NUM_ENVS = DEFAULT_NUM_ENVS
USE_TENSORBOARD = True
EVAL_DURING_TRAINING = False
EVAL_INTERVAL = 50_000
EVAL_EPISODES = 5


@dataclass
class TrainConfig:
    run_mode: str = "fresh_start"
    seed: Optional[int] = None
    num_seeds: int = DEFAULT_NUM_SEEDS
    num_envs: int = DEFAULT_NUM_ENVS
    seed_values: tuple[int, ...] = ()
    map_name: str = MAP_NAME
    agent_name: str = AGENT_NAME

    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    save_interval: int = DEFAULT_SAVE_INTERVAL
    log_interval: int = LOG_INTERVAL
    eval_interval: int = EVAL_INTERVAL
    eval_episodes: int = 5
    eval_during_training: bool = EVAL_DURING_TRAINING

    buffer_size: int = 5000
    batch_size: int = 16
    learn_start_episodes: int = 32
    train_updates_per_episode: int = 1

    gamma: float = 0.99
    learning_rate: float = 5e-4
    grad_norm_clip: float = 10.0
    target_update_interval: int = 200

    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_timesteps: int = 50_000

    hidden_dim: int = 64
    minimap_embed_dim: int = 64
    use_rnn: bool = False
    use_minimap: bool = True
    obs_agent_id: bool = True
    double_q: bool = True
    standardise_rewards: bool = True

    mixing_embed_dim: int = 32
    hypernet_layers: int = 2
    hypernet_embed: int = 64

    use_tensorboard: bool = USE_TENSORBOARD
    visualize: bool = False
    realtime: bool = False
    episode_limit: Optional[int] = None
    replay_dir: str = ""
    save_replay_episodes: int = 0
    resume_checkpoint: str = ""
    summary_window: int = 20
    save_replay_buffer: bool = True
    replay_save_max_episodes: int = 256
    replay_save_max_bytes: int = 128 * 1024 * 1024
    save_rng_state: bool = True

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_run_config():
    return TrainConfig(
        run_mode=RUN_MODE,
        seed=FRESH_START_SEED,
        seed_values=tuple(int(seed) for seed in FRESH_START_SEED_VALUES),
        num_seeds=NUM_SEEDS,
        num_envs=NUM_ENVS,
        map_name=MAP_NAME,
        agent_name=AGENT_NAME,
        total_timesteps=TOTAL_TIMESTEPS,
        save_interval=SAVE_INTERVAL,
        log_interval=LOG_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        eval_episodes=EVAL_EPISODES,
        eval_during_training=EVAL_DURING_TRAINING,
        use_tensorboard=USE_TENSORBOARD,
    )


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


def get_cuda_memory_stats(device: str):
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def generate_random_seeds(num_seeds: int):
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


def resolve_seeds(config: TrainConfig):
    if config.seed_values:
        return tuple(int(seed) for seed in config.seed_values)
    if config.seed is not None:
        return (int(config.seed),)
    return generate_random_seeds(config.num_seeds)


def next_multiple(current: int, interval: int):
    if interval <= 0:
        return current
    return ((current // interval) + 1) * interval


def step_label(env_steps: int):
    if env_steps % 1_000_000 == 0:
        return f"{env_steps // 1_000_000}M"
    if env_steps % 1_000 == 0:
        return f"{env_steps // 1_000}K"
    return str(env_steps)


def get_agent_save_root(config: TrainConfig):
    return PROJECT_ROOT / "Agents" / "Qmix" / config.map_name / "saved_models" / config.agent_name


def get_agent_tb_root(config: TrainConfig):
    return PROJECT_ROOT / "tb_logs" / "Qmix" / config.map_name / config.agent_name


def get_seed_output_dirs(config: TrainConfig, seed: int):
    save_dir = get_agent_save_root(config) / f"seed_{seed}"
    tb_log_dir = get_agent_tb_root(config) / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, tb_log_dir


def write_seed_manifest(config: TrainConfig, seeds):
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent_name": config.agent_name,
        "map_name": config.map_name,
        "run_mode": config.run_mode,
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "num_envs": config.num_envs,
        "total_timesteps": config.total_timesteps,
        "save_interval": config.save_interval,
        "batch_size": config.batch_size,
        "buffer_size": config.buffer_size,
        "device": config.device,
    }
    save_root = get_agent_save_root(config)
    save_root.mkdir(parents=True, exist_ok=True)

    latest_manifest_path = save_root / "latest_run_manifest.json"
    dated_manifest_path = save_root / f"run_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    for path in (latest_manifest_path, dated_manifest_path):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    return latest_manifest_path


def is_final_checkpoint_name(config: TrainConfig, checkpoint_name: str):
    return checkpoint_name == f"{config.agent_name}_final.pt"


def parse_checkpoint_steps(config: TrainConfig, checkpoint_path):
    checkpoint_name = Path(checkpoint_path).name
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
    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers[scale]


def checkpoint_sort_key(config: TrainConfig, checkpoint_path):
    parsed_steps = parse_checkpoint_steps(config, checkpoint_path)
    if parsed_steps is None:
        return None
    return (parsed_steps, Path(checkpoint_path).stat().st_mtime)


def collect_seed_checkpoints(config: TrainConfig, seed: int):
    save_dir = get_agent_save_root(config) / f"seed_{seed}"
    if not save_dir.is_dir():
        return []

    checkpoint_paths = []
    for entry in save_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".pt":
            continue
        if is_final_checkpoint_name(config, entry.name):
            continue
        if checkpoint_sort_key(config, entry) is None:
            continue
        checkpoint_paths.append(entry)
    return checkpoint_paths


def seed_dir_has_final(config: TrainConfig, seed_dir: Path):
    for entry in seed_dir.iterdir():
        if entry.is_file() and is_final_checkpoint_name(config, entry.name):
            return True
    return False


def load_latest_seed_manifest(config: TrainConfig):
    manifest_path = get_agent_save_root(config) / "latest_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No seed manifest found at {manifest_path}. Run fresh_start first to create it."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    raw_seeds = manifest.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise RuntimeError(
            f"Seed manifest {manifest_path} is missing a non-empty 'seeds' list."
        )

    seeds = []
    seen = set()
    for raw_seed in raw_seeds:
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Seed manifest {manifest_path} contains a non-integer seed: {raw_seed!r}"
            ) from exc
        if seed in seen:
            raise RuntimeError(
                f"Seed manifest {manifest_path} contains duplicate seed {seed}."
            )
        seeds.append(seed)
        seen.add(seed)

    manifest["seeds"] = tuple(seeds)
    return manifest_path, manifest


def describe_seed_progress(config: TrainConfig, seed: int):
    save_dir = get_agent_save_root(config) / f"seed_{seed}"
    checkpoint_paths = collect_seed_checkpoints(config, seed)
    checkpoint_path = (
        max(checkpoint_paths, key=lambda path: checkpoint_sort_key(config, path))
        if checkpoint_paths
        else None
    )
    return {
        "seed": seed,
        "save_dir": save_dir,
        "has_final": save_dir.is_dir() and seed_dir_has_final(config, save_dir),
        "checkpoint_path": checkpoint_path,
    }


def resolve_resume_plan(config: TrainConfig):
    manifest_path, manifest = load_latest_seed_manifest(config)
    seed_states = [describe_seed_progress(config, seed) for seed in manifest["seeds"]]
    pending_states = [state for state in seed_states if not state["has_final"]]

    if not pending_states:
        raise FileNotFoundError(
            "No unfinished seed found. Every seed from latest_run_manifest.json already has a _final checkpoint."
        )

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "seed_states": seed_states,
        "pending_states": pending_states,
    }


def normalize_config(config: TrainConfig):
    if config.run_mode not in {"fresh_start", "load_last_checkpoint"}:
        raise ValueError(
            f"Invalid RUN_MODE: {config.run_mode!r}. Use 'fresh_start' or 'load_last_checkpoint'."
        )
    if config.seed is not None and config.seed_values:
        raise ValueError("Use either seed or seed_values, not both.")
    if config.num_envs < 1:
        raise ValueError("NUM_ENVS must be at least 1")
    if config.num_seeds < 1:
        raise ValueError("NUM_SEEDS must be at least 1")
    if config.total_timesteps < 1:
        raise ValueError("TOTAL_TIMESTEPS must be at least 1")
    if config.save_interval < 1:
        raise ValueError("SAVE_INTERVAL must be at least 1")
    if config.save_interval > config.total_timesteps:
        raise ValueError("SAVE_INTERVAL cannot exceed TOTAL_TIMESTEPS")
    if config.log_interval < 1:
        raise ValueError("LOG_INTERVAL must be at least 1")
    if config.eval_during_training:
        if config.eval_interval < 1:
            raise ValueError("EVAL_INTERVAL must be at least 1 when eval_during_training=True")
        if config.eval_episodes < 1:
            raise ValueError("EVAL_EPISODES must be at least 1 when eval_during_training=True")
    else:
        config.eval_interval = 0
        config.eval_episodes = 0
    config.replay_save_max_episodes = max(0, int(config.replay_save_max_episodes))
    config.replay_save_max_bytes = max(0, int(config.replay_save_max_bytes))
    return config


def capture_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    if not state:
        return

    python_state = state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(tuple(numpy_state))

    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state.cpu())

    cuda_states = state.get("torch_cuda_random_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in cuda_states])


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _make_env_payload(env: TwoBridgeEnv, obs):
    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "state": np.asarray(env.get_state(), dtype=np.float32),
        "minimap": np.asarray(env.get_minimap(), dtype=np.uint8),
        "avail_actions": np.asarray(env.get_avail_actions(), dtype=np.float32),
    }


def qmix_env_worker(remote, parent_remote, rank: int, env_kwargs: dict):
    parent_remote.close()
    worker_tmp_dir = tempfile.mkdtemp(prefix=f"tbm-qmix-worker-{rank}-")
    cleanup = lambda path=worker_tmp_dir: shutil.rmtree(path, ignore_errors=True)
    atexit.register(cleanup)

    try:
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = worker_tmp_dir

        time.sleep(0.5 * rank)
        env = TwoBridgeEnv(**env_kwargs)

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
        error_message = traceback.format_exc()
        try:
            remote.send({"__worker_error__": error_message})
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass
        cleanup()


class ParallelQMixEnvBatch:
    def __init__(self, num_envs: int, base_seed: int, env_kwargs: dict):
        self.num_envs = int(num_envs)
        self.closed = False
        self.ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []

        for rank in range(self.num_envs):
            parent_remote, worker_remote = self.ctx.Pipe()
            worker_kwargs = dict(env_kwargs)
            worker_kwargs["seed"] = int(base_seed + rank)
            process = self.ctx.Process(
                target=qmix_env_worker,
                args=(worker_remote, parent_remote, rank, worker_kwargs),
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
        batch_count = values.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
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


class SharedQAgent(nn.Module):
    """
    Small shared agent network adapted from EPyMARL's RNNAgent.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, use_rnn: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if self.use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size: int, device: str):
        return self.fc1.weight.new_zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor):
        x = F.relu(self.fc1(inputs))
        if self.use_rnn:
            hidden_state = self.rnn(x, hidden_state)
        else:
            hidden_state = F.relu(self.rnn(x))
        q_values = self.fc2(hidden_state)
        return q_values, hidden_state


class QMixer(nn.Module):
    """
    Mixer adapted from EPyMARL's QMixer.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int,
        hypernet_layers: int,
        hypernet_embed: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = int(state_dim)
        self.embed_dim = int(mixing_embed_dim)

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise ValueError("Only 1 or 2 hypernet layers are supported.")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.value_fn = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        batch_size = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states)).view(-1, self.embed_dim, 1)
        v = self.value_fn(states).view(-1, 1, 1)
        q_total = torch.bmm(hidden, w_final) + v
        return q_total.view(batch_size, -1, 1)


class EpisodeReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)

    def __len__(self):
        return len(self.episodes)

    def add(self, episode: dict):
        self.episodes.append(episode)

    @staticmethod
    def _estimate_value_bytes(value):
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(EpisodeReplayBuffer._estimate_value_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(EpisodeReplayBuffer._estimate_value_bytes(item) for item in value)
        return 0

    def snapshot(
        self,
        max_episodes: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ):
        episodes = list(self.episodes)
        total_available = len(episodes)

        if max_episodes is not None:
            max_episodes = max(0, int(max_episodes))
            if max_episodes == 0:
                return {
                    "episodes": [],
                    "saved_episodes": 0,
                    "available_episodes": total_available,
                    "total_bytes": 0,
                    "truncated": total_available > 0,
                }
            episodes = episodes[-max_episodes:]

        if max_bytes is None or int(max_bytes) <= 0:
            total_bytes = sum(self._estimate_value_bytes(episode) for episode in episodes)
            return {
                "episodes": episodes,
                "saved_episodes": len(episodes),
                "available_episodes": total_available,
                "total_bytes": total_bytes,
                "truncated": len(episodes) < total_available,
            }

        max_bytes = int(max_bytes)
        selected = deque()
        total_bytes = 0
        skipped_for_size = 0

        for episode in reversed(episodes):
            episode_bytes = self._estimate_value_bytes(episode)
            if episode_bytes > max_bytes:
                skipped_for_size += 1
                continue
            if total_bytes + episode_bytes > max_bytes:
                break
            selected.appendleft(episode)
            total_bytes += episode_bytes

        saved_episodes = len(selected)
        return {
            "episodes": list(selected),
            "saved_episodes": saved_episodes,
            "available_episodes": total_available,
            "total_bytes": total_bytes,
            "truncated": (saved_episodes < total_available) or (skipped_for_size > 0),
        }

    def load_snapshot(self, episodes):
        if episodes is None:
            self.episodes = deque(maxlen=self.capacity)
            return
        self.episodes = deque(episodes, maxlen=self.capacity)

    def sample(self, batch_size: int, device: str):
        sampled = random.sample(self.episodes, batch_size)
        max_t = max(ep["actions"].shape[0] for ep in sampled)

        obs_shape = sampled[0]["obs"].shape[2]
        state_shape = sampled[0]["state"].shape[1]
        minimap_shape = sampled[0]["minimap"].shape[1:]
        n_agents = sampled[0]["obs"].shape[1]
        n_actions = sampled[0]["avail_actions"].shape[2]

        obs = torch.zeros((batch_size, max_t + 1, n_agents, obs_shape), dtype=torch.float32, device=device)
        state = torch.zeros((batch_size, max_t + 1, state_shape), dtype=torch.float32, device=device)
        minimap = torch.zeros((batch_size, max_t + 1, *minimap_shape), dtype=torch.uint8, device=device)
        avail_actions = torch.zeros(
            (batch_size, max_t + 1, n_agents, n_actions), dtype=torch.float32, device=device
        )
        actions = torch.zeros((batch_size, max_t, n_agents, 1), dtype=torch.long, device=device)
        rewards = torch.zeros((batch_size, max_t, 1), dtype=torch.float32, device=device)
        terminated = torch.zeros((batch_size, max_t, 1), dtype=torch.float32, device=device)
        filled = torch.zeros((batch_size, max_t, 1), dtype=torch.float32, device=device)

        for batch_idx, episode in enumerate(sampled):
            episode_t = episode["actions"].shape[0]
            obs[batch_idx, : episode_t + 1] = torch.as_tensor(episode["obs"], dtype=torch.float32, device=device)
            state[batch_idx, : episode_t + 1] = torch.as_tensor(episode["state"], dtype=torch.float32, device=device)
            minimap[batch_idx, : episode_t + 1] = torch.as_tensor(
                episode["minimap"], dtype=torch.uint8, device=device
            )
            avail_actions[batch_idx, : episode_t + 1] = torch.as_tensor(
                episode["avail_actions"], dtype=torch.float32, device=device
            )
            actions[batch_idx, :episode_t, :, 0] = torch.as_tensor(
                episode["actions"], dtype=torch.long, device=device
            )
            rewards[batch_idx, :episode_t, 0] = torch.as_tensor(
                episode["reward"], dtype=torch.float32, device=device
            )
            terminated[batch_idx, :episode_t, 0] = torch.as_tensor(
                episode["terminated"], dtype=torch.float32, device=device
            )
            filled[batch_idx, :episode_t, 0] = 1.0

        return {
            "obs": obs,
            "state": state,
            "minimap": minimap,
            "avail_actions": avail_actions,
            "actions": actions,
            "reward": rewards,
            "terminated": terminated,
            "filled": filled,
            "max_seq_length": max_t + 1,
        }


class QMixTrainer:
    def __init__(self, config: TrainConfig, seed: int):
        self.config = config
        self.seed = int(seed)
        self.device = config.device
        self.num_envs = int(config.num_envs)

        self.save_dir, self.tb_log_dir = get_seed_output_dirs(config, self.seed)

        self.writer = None
        if config.use_tensorboard:
            if SummaryWriter is None:
                print("TensorBoard is unavailable in this interpreter. Disabling tensorboard logging.")
            else:
                self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))

        train_env_kwargs = {
            "map_name": config.map_name,
            "episode_limit": config.episode_limit,
            "visualize": config.visualize,
            "realtime": config.realtime,
            "replay_dir": config.replay_dir,
            "save_replay_episodes": config.save_replay_episodes,
        }
        if self.num_envs == 1:
            self.train_env = TwoBridgeEnv(
                seed=self.seed,
                **train_env_kwargs,
            )
        else:
            self.train_env = ParallelQMixEnvBatch(
                num_envs=self.num_envs,
                base_seed=self.seed,
                env_kwargs=train_env_kwargs,
            )
        self.eval_env = None
        if config.eval_during_training:
            self.eval_env = TwoBridgeEnv(
                map_name=config.map_name,
                seed=self.seed + 10_000,
                episode_limit=config.episode_limit,
                visualize=False,
                realtime=False,
                replay_dir=config.replay_dir,
                save_replay_episodes=0,
            )

        env_info = (
            self.train_env.get_env_info()
            if self.num_envs == 1
            else self.train_env.env_info
        )
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_dim = env_info["obs_shape"]
        self.state_dim = env_info["state_shape"]
        self.minimap_shape = tuple(env_info.get("minimap_shape", ()))
        self.episode_limit = env_info["episode_limit"]
        self.use_minimap = bool(config.use_minimap and self.minimap_shape)

        self.minimap_encoder = None
        self.target_minimap_encoder = None
        self.minimap_embed_dim = 0
        if self.use_minimap:
            self.minimap_embed_dim = int(config.minimap_embed_dim)
            self.minimap_encoder = MinimapEncoder(
                in_channels=self.minimap_shape[0],
                height=self.minimap_shape[1],
                width=self.minimap_shape[2],
                embed_dim=self.minimap_embed_dim,
            ).to(self.device)
            self.target_minimap_encoder = copy.deepcopy(self.minimap_encoder)

        self.agent_input_dim = (
            self.obs_dim
            + self.minimap_embed_dim
            + (self.n_agents if config.obs_agent_id else 0)
        )
        self.agent = SharedQAgent(
            input_dim=self.agent_input_dim,
            hidden_dim=config.hidden_dim,
            n_actions=self.n_actions,
            use_rnn=config.use_rnn,
        ).to(self.device)
        self.target_agent = copy.deepcopy(self.agent)

        self.mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=self.state_dim + self.minimap_embed_dim,
            mixing_embed_dim=config.mixing_embed_dim,
            hypernet_layers=config.hypernet_layers,
            hypernet_embed=config.hypernet_embed,
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = torch.optim.Adam(
            self.online_parameters(),
            lr=config.learning_rate,
        )
        self.reward_ms = RunningMeanStd(shape=(1,), device=self.device) if config.standardise_rewards else None
        self.replay_buffer = EpisodeReplayBuffer(config.buffer_size)

        self.env_steps = 0
        self.episode_count = 0
        self.train_updates = 0
        self.next_save_step = config.save_interval
        self.next_log_step = config.log_interval
        self.next_eval_step = config.eval_interval if config.eval_during_training else None
        self.recent_returns = deque(maxlen=config.summary_window)
        self.recent_lengths = deque(maxlen=config.summary_window)
        self.recent_outcomes = deque(maxlen=config.summary_window)

        self._write_run_config()
        self._maybe_load_checkpoint()

    def close(self):
        self.train_env.close()
        if self.eval_env is not None:
            self.eval_env.close()
        if self.writer is not None:
            self.writer.close()
        release_training_memory()

    def train(self):
        print(f"Using device: {self.device} | SEED={self.seed}")
        print(
            "QMIX setup | "
            f"num_envs={self.num_envs} | "
            f"n_agents={self.n_agents} | n_actions={self.n_actions} | "
            f"obs_dim={self.obs_dim} | state_dim={self.state_dim} | "
            f"minimap_shape={self.minimap_shape if self.use_minimap else None} | "
            f"episode_limit={self.episode_limit}"
        )
        print(f"Checkpoint dir: {self.save_dir}")
        if self.writer is not None:
            print(f"TensorBoard dir: {self.tb_log_dir}")
        if self.config.resume_checkpoint:
            print(f"Resume checkpoint: {self.config.resume_checkpoint}")

        if self.num_envs > 1:
            self.train_parallel()
            self.save_checkpoint("final")
            print(
                f"Training finished at env_steps={self.env_steps}, "
                f"episodes={self.episode_count}"
            )
            return

        while self.env_steps < self.config.total_timesteps:
            epsilon = self.current_epsilon(self.env_steps)
            episode = self.rollout_episode(self.train_env, epsilon=epsilon, test_mode=False)

            self.env_steps += episode["length"]
            self.record_episode(episode)
            self.run_periodic_tasks()

        self.save_checkpoint("final")
        print(f"Training finished at env_steps={self.env_steps}, episodes={self.episode_count}")

    def record_episode(self, episode: dict):
        self.episode_count += 1
        self.replay_buffer.add(episode["batch"])

        self.recent_returns.append(episode["return"])
        self.recent_lengths.append(episode["length"])
        self.recent_outcomes.append(episode["result"])

        self.log_episode(episode, prefix="train")

        if len(self.replay_buffer) >= max(self.config.batch_size, self.config.learn_start_episodes):
            for _ in range(self.config.train_updates_per_episode):
                train_stats = self.train_step()
                if train_stats is not None:
                    self.train_updates += 1
                    if self.train_updates % self.config.target_update_interval == 0:
                        self.update_targets()
                    self.log_train_step(train_stats)

    def run_periodic_tasks(self):
        while self.env_steps >= self.next_log_step:
            self.print_progress()
            self.next_log_step += self.config.log_interval

        if self.config.eval_during_training:
            while self.env_steps >= self.next_eval_step:
                eval_stats = self.evaluate_policy()
                self.log_eval(eval_stats)
                self.next_eval_step += self.config.eval_interval

        while self.env_steps >= self.next_save_step:
            self.save_checkpoint(step_label(self.next_save_step))
            self.next_save_step += self.config.save_interval

    def create_episode_tracker(self, obs, state, minimap, avail_actions):
        return {
            "obs_seq": [obs.copy()],
            "state_seq": [state.copy()],
            "minimap_seq": [minimap.copy()],
            "avail_seq": [avail_actions.copy()],
            "action_seq": [],
            "reward_seq": [],
            "terminated_seq": [],
            "reward_component_sums": Counter(),
            "episode_return": 0.0,
        }

    def append_episode_transition(
        self,
        tracker: dict,
        actions,
        reward: float,
        done: bool,
        info: dict,
        next_obs,
        next_state,
        next_minimap,
        next_avail_actions,
    ):
        tracker["action_seq"].append(np.asarray(actions, dtype=np.int64).copy())
        tracker["reward_seq"].append(float(reward))
        tracker["terminated_seq"].append(float(done and not info.get("episode_limit", False)))
        tracker["episode_return"] += float(reward)

        for key, value in info.get("rew", {}).items():
            tracker["reward_component_sums"][key] += float(value)

        tracker["obs_seq"].append(np.asarray(next_obs, dtype=np.float32).copy())
        tracker["state_seq"].append(np.asarray(next_state, dtype=np.float32).copy())
        tracker["minimap_seq"].append(np.asarray(next_minimap, dtype=np.uint8).copy())
        tracker["avail_seq"].append(np.asarray(next_avail_actions, dtype=np.float32).copy())

    def finalize_episode_tracker(self, tracker: dict, final_info: dict):
        length = len(tracker["action_seq"])
        component_means = {
            key: value / max(length, 1)
            for key, value in tracker["reward_component_sums"].items()
        }
        batch = {
            "obs": np.asarray(tracker["obs_seq"], dtype=np.float32),
            "state": np.asarray(tracker["state_seq"], dtype=np.float32),
            "minimap": np.asarray(tracker["minimap_seq"], dtype=np.uint8),
            "avail_actions": np.asarray(tracker["avail_seq"], dtype=np.float32),
            "actions": np.asarray(tracker["action_seq"], dtype=np.int64),
            "reward": np.asarray(tracker["reward_seq"], dtype=np.float32),
            "terminated": np.asarray(tracker["terminated_seq"], dtype=np.float32),
        }
        return {
            "batch": batch,
            "return": tracker["episode_return"],
            "length": length,
            "result": final_info.get("result"),
            "reward_components": component_means,
            "final_info": final_info,
        }

    def train_parallel(self):
        reset_results = self.train_env.reset()
        current_obs = np.zeros(
            (self.num_envs, self.n_agents, self.obs_dim), dtype=np.float32
        )
        current_state = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        current_minimap = np.zeros(
            (self.num_envs, *self.minimap_shape), dtype=np.uint8
        )
        current_avail = np.zeros(
            (self.num_envs, self.n_agents, self.n_actions), dtype=np.float32
        )
        episode_trackers = [None] * self.num_envs

        for index, payload in reset_results:
            current_obs[index] = payload["obs"]
            current_state[index] = payload["state"]
            current_minimap[index] = payload["minimap"]
            current_avail[index] = payload["avail_actions"]
            episode_trackers[index] = self.create_episode_tracker(
                payload["obs"],
                payload["state"],
                payload["minimap"],
                payload["avail_actions"],
            )

        hidden = self.agent.init_hidden(
            self.num_envs * self.n_agents, self.device
        ).view(self.num_envs, self.n_agents, -1)

        while self.env_steps < self.config.total_timesteps:
            epsilon = self.current_epsilon(self.env_steps)
            actions_batch, next_hidden = self.select_actions_batch(
                obs_batch=current_obs,
                minimap_batch=current_minimap,
                avail_actions_batch=current_avail,
                epsilon=epsilon,
                hidden_batch=hidden,
                greedy=False,
            )
            step_results = self.train_env.step(actions_batch)
            hidden = next_hidden
            self.env_steps += len(step_results)

            completed_indices = []
            for index, payload in step_results:
                done = bool(payload["terminated"] or payload["truncated"])
                self.append_episode_transition(
                    episode_trackers[index],
                    actions_batch[index],
                    payload["reward"],
                    done,
                    payload["info"],
                    payload["obs"],
                    payload["state"],
                    payload["minimap"],
                    payload["avail_actions"],
                )
                current_obs[index] = payload["obs"]
                current_state[index] = payload["state"]
                current_minimap[index] = payload["minimap"]
                current_avail[index] = payload["avail_actions"]

                if done:
                    episode = self.finalize_episode_tracker(
                        episode_trackers[index], payload["info"]
                    )
                    self.record_episode(episode)
                    completed_indices.append(index)

            self.run_periodic_tasks()
            if self.env_steps >= self.config.total_timesteps:
                break

            if completed_indices:
                reset_results = self.train_env.reset(completed_indices)
                for index, payload in reset_results:
                    current_obs[index] = payload["obs"]
                    current_state[index] = payload["state"]
                    current_minimap[index] = payload["minimap"]
                    current_avail[index] = payload["avail_actions"]
                    episode_trackers[index] = self.create_episode_tracker(
                        payload["obs"],
                        payload["state"],
                        payload["minimap"],
                        payload["avail_actions"],
                    )
                    hidden[index] = self.agent.init_hidden(self.n_agents, self.device)

    def rollout_episode(self, env: TwoBridgeEnv, epsilon: float, test_mode: bool):
        obs, _ = env.reset()
        state = env.get_state()
        minimap = env.get_minimap()
        avail_actions = env.get_avail_actions()
        hidden = self.agent.init_hidden(env.n_agents, self.device)
        tracker = self.create_episode_tracker(obs, state, minimap, avail_actions)
        done = False
        final_info = {"result": None}

        while not done:
            actions, hidden = self.select_actions(
                obs=obs,
                minimap=minimap,
                avail_actions=avail_actions,
                epsilon=0.0 if test_mode else epsilon,
                hidden=hidden,
                greedy=test_mode,
            )
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            obs = next_obs
            state = env.get_state()
            minimap = env.get_minimap()
            avail_actions = env.get_avail_actions()
            self.append_episode_transition(
                tracker,
                actions,
                reward,
                done,
                info,
                obs,
                state,
                minimap,
                avail_actions,
            )
            final_info = info

        return self.finalize_episode_tracker(tracker, final_info)

    def train_step(self):
        batch = self.replay_buffer.sample(self.config.batch_size, device=self.device)
        rewards = batch["reward"]
        actions = batch["actions"]
        terminated = batch["terminated"]
        filled = batch["filled"]
        avail_actions = batch["avail_actions"]
        obs = batch["obs"]
        state = batch["state"]
        minimap = batch["minimap"]

        if self.reward_ms is not None:
            valid_rewards = rewards[filled.bool()]
            if valid_rewards.numel() > 0:
                self.reward_ms.update(valid_rewards.view(-1, 1))
                rewards = (rewards - self.reward_ms.mean.view(1, 1, 1)) / torch.sqrt(
                    self.reward_ms.var.view(1, 1, 1) + 1e-8
                )

        with torch.no_grad():
            target_minimap_embed = self.encode_minimap_sequence(
                self.target_minimap_encoder, minimap
            )
            target_mac_out = self.forward_sequence(
                self.target_agent, obs, target_minimap_embed
            )
            target_mac_out = target_mac_out[:, 1:]
            target_mac_out[avail_actions[:, 1:] == 0] = -1e9

        live_minimap_embed = self.encode_minimap_sequence(self.minimap_encoder, minimap)
        mac_out = self.forward_sequence(self.agent, obs, live_minimap_embed)
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        chosen_qtot = self.mixer(
            chosen_action_qvals,
            self.build_mixer_state(state[:, :-1], live_minimap_embed[:, :-1]),
        )

        if self.config.double_q:
            live_q = mac_out.detach().clone()
            live_q[avail_actions == 0] = -1e9
            cur_max_actions = live_q[:, 1:].max(dim=3, keepdim=True)[1]
        else:
            cur_max_actions = None

        with torch.no_grad():
            if cur_max_actions is not None:
                target_max_qvals = torch.gather(target_mac_out, dim=3, index=cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            target_qtot = self.target_mixer(
                target_max_qvals,
                self.build_mixer_state(state[:, 1:], target_minimap_embed[:, 1:]),
            )

        targets = rewards + self.config.gamma * (1.0 - terminated) * target_qtot
        td_error = chosen_qtot - targets
        masked_td_error = td_error * filled
        loss = (masked_td_error.pow(2).sum()) / filled.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_parameters(),
            self.config.grad_norm_clip,
        )
        self.optimizer.step()

        mask_elems = filled.sum().item()
        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "td_error_abs": float(masked_td_error.abs().sum().item() / max(mask_elems, 1.0)),
            "q_taken_mean": float((chosen_qtot * filled).sum().item() / max(mask_elems, 1.0)),
            "target_mean": float((targets * filled).sum().item() / max(mask_elems, 1.0)),
        }

    def forward_sequence(
        self,
        agent_net: SharedQAgent,
        obs_seq: torch.Tensor,
        minimap_embed_seq: Optional[torch.Tensor],
    ):
        batch_size, seq_length, _, _ = obs_seq.shape
        hidden = agent_net.init_hidden(batch_size * self.n_agents, self.device)
        outputs = []

        for t in range(seq_length):
            inputs = self.build_agent_inputs(
                obs_seq[:, t],
                None if minimap_embed_seq is None else minimap_embed_seq[:, t],
            )
            q_values, hidden = agent_net(inputs, hidden)
            outputs.append(q_values.view(batch_size, self.n_agents, self.n_actions))

        return torch.stack(outputs, dim=1)

    def build_agent_inputs(
        self, obs_t: torch.Tensor, minimap_embed_t: Optional[torch.Tensor] = None
    ):
        batch_size = obs_t.size(0)
        pieces = [obs_t]
        if minimap_embed_t is not None:
            minimap_features = minimap_embed_t.unsqueeze(1).expand(-1, self.n_agents, -1)
            pieces.append(minimap_features)
        if self.config.obs_agent_id:
            agent_ids = torch.eye(self.n_agents, device=obs_t.device).unsqueeze(0).expand(batch_size, -1, -1)
            pieces.append(agent_ids)
        return torch.cat(pieces, dim=-1).reshape(batch_size * self.n_agents, -1)

    def select_actions_batch(
        self,
        obs_batch,
        minimap_batch,
        avail_actions_batch,
        epsilon: float,
        hidden_batch: torch.Tensor,
        greedy: bool,
    ):
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        avail_tensor = torch.as_tensor(
            avail_actions_batch, dtype=torch.bool, device=self.device
        )
        minimap_tensor = torch.as_tensor(
            minimap_batch, dtype=torch.uint8, device=self.device
        )
        minimap_embed = self.encode_minimap(self.minimap_encoder, minimap_tensor)
        inputs = self.build_agent_inputs(obs_tensor, minimap_embed)

        batch_size = obs_tensor.size(0)
        hidden_flat = hidden_batch.reshape(batch_size * self.n_agents, -1)
        q_values, next_hidden = self.agent(inputs, hidden_flat)
        q_values = q_values.view(batch_size, self.n_agents, self.n_actions)
        next_hidden = next_hidden.view(batch_size, self.n_agents, -1)

        masked_q = q_values.clone()
        masked_q[~avail_tensor] = -1e9
        greedy_actions = masked_q.argmax(dim=2)

        if greedy:
            return (
                greedy_actions.detach().cpu().numpy().astype(np.int64),
                next_hidden.detach(),
            )

        chosen = np.zeros((batch_size, self.n_agents), dtype=np.int64)
        for env_idx in range(batch_size):
            for agent_id in range(self.n_agents):
                valid_actions = torch.nonzero(
                    avail_tensor[env_idx, agent_id], as_tuple=False
                ).squeeze(-1)
                if valid_actions.numel() == 0:
                    chosen[env_idx, agent_id] = 0
                    continue

                if random.random() < epsilon:
                    random_idx = random.randrange(valid_actions.numel())
                    chosen[env_idx, agent_id] = int(valid_actions[random_idx].item())
                else:
                    chosen[env_idx, agent_id] = int(
                        greedy_actions[env_idx, agent_id].item()
                    )

        return chosen, next_hidden.detach()

    def select_actions(
        self,
        obs,
        minimap,
        avail_actions,
        epsilon: float,
        hidden: torch.Tensor,
        greedy: bool,
    ):
        actions_batch, next_hidden_batch = self.select_actions_batch(
            obs_batch=np.expand_dims(obs, axis=0),
            minimap_batch=np.expand_dims(minimap, axis=0),
            avail_actions_batch=np.expand_dims(avail_actions, axis=0),
            epsilon=epsilon,
            hidden_batch=hidden.unsqueeze(0),
            greedy=greedy,
        )
        return actions_batch[0], next_hidden_batch[0]

    def current_epsilon(self, env_steps: int):
        if self.config.epsilon_anneal_timesteps <= 0:
            return self.config.epsilon_finish

        frac = min(float(env_steps) / float(self.config.epsilon_anneal_timesteps), 1.0)
        return self.config.epsilon_start + frac * (
            self.config.epsilon_finish - self.config.epsilon_start
        )

    def update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        if self.use_minimap:
            self.target_minimap_encoder.load_state_dict(self.minimap_encoder.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def online_parameters(self):
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        if self.use_minimap:
            params += list(self.minimap_encoder.parameters())
        return params

    def encode_minimap(self, encoder: Optional[MinimapEncoder], minimap_batch: torch.Tensor):
        if encoder is None:
            return None
        return encoder(minimap_batch)

    def encode_minimap_sequence(
        self, encoder: Optional[MinimapEncoder], minimap_seq: torch.Tensor
    ):
        if encoder is None:
            return None
        batch_size, seq_length, channels, height, width = minimap_seq.shape
        flat = minimap_seq.reshape(batch_size * seq_length, channels, height, width)
        encoded = encoder(flat)
        return encoded.view(batch_size, seq_length, -1)

    def build_mixer_state(
        self, state_seq: torch.Tensor, minimap_embed_seq: Optional[torch.Tensor]
    ):
        if minimap_embed_seq is None:
            return state_seq
        return torch.cat([state_seq, minimap_embed_seq], dim=-1)

    def evaluate_policy(self):
        returns = []
        lengths = []
        outcomes = Counter()

        for _ in range(self.config.eval_episodes):
            episode = self.rollout_episode(self.eval_env, epsilon=0.0, test_mode=True)
            returns.append(episode["return"])
            lengths.append(episode["length"])
            outcomes[episode["result"]] += 1

        return {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "length_mean": float(np.mean(lengths)),
            "outcomes": dict(outcomes),
        }

    def log_episode(self, episode: dict, prefix: str):
        if self.writer is None:
            return

        self.writer.add_scalar(f"{prefix}/return", episode["return"], self.env_steps)
        self.writer.add_scalar(f"{prefix}/episode_length", episode["length"], self.env_steps)
        self.writer.add_scalar(f"{prefix}/epsilon", self.current_epsilon(self.env_steps), self.env_steps)
        for key, value in episode["reward_components"].items():
            self.writer.add_scalar(f"{prefix}_rew/{key}", value, self.env_steps)

    def log_train_step(self, stats: dict):
        if self.writer is None:
            return
        for key, value in stats.items():
            self.writer.add_scalar(f"loss/{key}", value, self.env_steps)

    def log_eval(self, stats: dict):
        print(
            "Eval | "
            f"env_steps={self.env_steps} | "
            f"return_mean={stats['return_mean']:.3f} | "
            f"return_std={stats['return_std']:.3f} | "
            f"length_mean={stats['length_mean']:.1f} | "
            f"outcomes={stats['outcomes']}"
        )
        if self.writer is None:
            return

        self.writer.add_scalar("eval/return_mean", stats["return_mean"], self.env_steps)
        self.writer.add_scalar("eval/return_std", stats["return_std"], self.env_steps)
        self.writer.add_scalar("eval/length_mean", stats["length_mean"], self.env_steps)
        for key, value in stats["outcomes"].items():
            self.writer.add_scalar(f"eval_outcomes/{key}", value, self.env_steps)

    def print_progress(self):
        mean_return = float(np.mean(self.recent_returns)) if self.recent_returns else 0.0
        mean_length = float(np.mean(self.recent_lengths)) if self.recent_lengths else 0.0
        outcome_counts = Counter(self.recent_outcomes)
        memory_stats = get_cuda_memory_stats(self.device)
        memory_text = ""
        if memory_stats is not None:
            memory_text = (
                f" | cuda_alloc_mb={memory_stats['allocated_mb']:.0f}"
                f" | cuda_reserved_mb={memory_stats['reserved_mb']:.0f}"
                f" | cuda_peak_mb={memory_stats['max_allocated_mb']:.0f}"
            )
        print(
            "Train | "
            f"env_steps={self.env_steps} | "
            f"episodes={self.episode_count} | "
            f"buffer={len(self.replay_buffer)} | "
            f"updates={self.train_updates} | "
            f"epsilon={self.current_epsilon(self.env_steps):.3f} | "
            f"mean_return={mean_return:.3f} | "
            f"mean_length={mean_length:.1f} | "
            f"recent_outcomes={dict(outcome_counts)}"
            f"{memory_text}"
        )
        if self.writer is not None and memory_stats is not None:
            self.writer.add_scalar("system/cuda_alloc_mb", memory_stats["allocated_mb"], self.env_steps)
            self.writer.add_scalar("system/cuda_reserved_mb", memory_stats["reserved_mb"], self.env_steps)
            self.writer.add_scalar("system/cuda_peak_mb", memory_stats["max_allocated_mb"], self.env_steps)

    def save_checkpoint(self, label: str):
        checkpoint = {
            "config": asdict(self.config),
            "seed": self.seed,
            "env_steps": self.env_steps,
            "episode_count": self.episode_count,
            "train_updates": self.train_updates,
            "agent_state_dict": self.agent.state_dict(),
            "target_agent_state_dict": self.target_agent.state_dict(),
            "mixer_state_dict": self.mixer.state_dict(),
            "target_mixer_state_dict": self.target_mixer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "recent_returns": list(self.recent_returns),
            "recent_lengths": list(self.recent_lengths),
            "recent_outcomes": list(self.recent_outcomes),
        }
        if self.config.save_rng_state:
            checkpoint["rng_state"] = capture_rng_state()
        if self.use_minimap:
            checkpoint["minimap_encoder_state_dict"] = self.minimap_encoder.state_dict()
            checkpoint["target_minimap_encoder_state_dict"] = self.target_minimap_encoder.state_dict()
        if self.reward_ms is not None:
            checkpoint["reward_ms"] = {
                "mean": self.reward_ms.mean.detach().cpu(),
                "var": self.reward_ms.var.detach().cpu(),
                "count": self.reward_ms.count,
            }

        checkpoint_path = self.save_dir / f"{self.config.agent_name}_{label}.pt"
        torch.save(checkpoint, checkpoint_path)
        self._save_replay_buffer_checkpoint(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        release_training_memory()

    def _maybe_load_checkpoint(self):
        if not self.config.resume_checkpoint:
            return

        checkpoint_path = Path(self.config.resume_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.target_agent.load_state_dict(checkpoint["target_agent_state_dict"])
        if self.use_minimap:
            self.minimap_encoder.load_state_dict(checkpoint["minimap_encoder_state_dict"])
            self.target_minimap_encoder.load_state_dict(
                checkpoint["target_minimap_encoder_state_dict"]
            )
        self.mixer.load_state_dict(checkpoint["mixer_state_dict"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.episode_count = int(checkpoint.get("episode_count", 0))
        self.train_updates = int(checkpoint.get("train_updates", 0))

        reward_ms = checkpoint.get("reward_ms")
        if reward_ms is not None and self.reward_ms is not None:
            self.reward_ms.mean.copy_(reward_ms["mean"].to(self.device))
            self.reward_ms.var.copy_(reward_ms["var"].to(self.device))
            self.reward_ms.count = reward_ms["count"]

        self.recent_returns = deque(
            checkpoint.get("recent_returns", []),
            maxlen=self.config.summary_window,
        )
        self.recent_lengths = deque(
            checkpoint.get("recent_lengths", []),
            maxlen=self.config.summary_window,
        )
        self.recent_outcomes = deque(
            checkpoint.get("recent_outcomes", []),
            maxlen=self.config.summary_window,
        )

        self._maybe_load_replay_buffer_checkpoint(checkpoint_path)
        restore_rng_state(checkpoint.get("rng_state"))

        self.next_save_step = next_multiple(self.env_steps, self.config.save_interval)
        self.next_log_step = next_multiple(self.env_steps, self.config.log_interval)
        if self.config.eval_during_training:
            self.next_eval_step = next_multiple(self.env_steps, self.config.eval_interval)
        print(f"Resumed from checkpoint: {checkpoint_path}")

    def _get_replay_checkpoint_path(self, checkpoint_path: Path):
        return checkpoint_path.with_suffix(".replay.pt")

    def _save_replay_buffer_checkpoint(self, checkpoint_path: Path):
        replay_path = self._get_replay_checkpoint_path(checkpoint_path)
        if not self.config.save_replay_buffer:
            if replay_path.exists():
                replay_path.unlink(missing_ok=True)
            return

        snapshot = self.replay_buffer.snapshot(
            max_episodes=self.config.replay_save_max_episodes,
            max_bytes=self.config.replay_save_max_bytes,
        )
        replay_payload = {
            "buffer_capacity": self.replay_buffer.capacity,
            "saved_episodes": snapshot["saved_episodes"],
            "available_episodes": snapshot["available_episodes"],
            "total_bytes": snapshot["total_bytes"],
            "truncated": snapshot["truncated"],
            "episodes": snapshot["episodes"],
        }
        torch.save(replay_payload, replay_path)
        if replay_payload["available_episodes"] and replay_payload["truncated"]:
            print(
                "Replay buffer checkpoint capped | "
                f"saved_episodes={replay_payload['saved_episodes']} | "
                f"available_episodes={replay_payload['available_episodes']} | "
                f"bytes={replay_payload['total_bytes']}"
            )

    def _maybe_load_replay_buffer_checkpoint(self, checkpoint_path: Path):
        if not self.config.save_replay_buffer:
            return

        replay_path = self._get_replay_checkpoint_path(checkpoint_path)
        if not replay_path.is_file():
            print(f"Replay buffer checkpoint missing: {replay_path}")
            return

        replay_payload = safe_torch_load(replay_path, map_location="cpu")
        replay_episodes = replay_payload.get("episodes", [])
        self.replay_buffer.load_snapshot(replay_episodes)
        print(
            "Replay buffer restored | "
            f"loaded_episodes={len(self.replay_buffer)} | "
            f"saved_episodes={replay_payload.get('saved_episodes', len(replay_episodes))} | "
            f"available_episodes={replay_payload.get('available_episodes', len(replay_episodes))}"
        )

    def _write_run_config(self):
        config_path = self.save_dir / "run_config.json"
        payload = asdict(self.config)
        payload["seed"] = self.seed
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def train_for_seed(config: TrainConfig, seed: int):
    set_global_seeds(seed)
    seed_start_time = time.perf_counter()
    seed_config = copy.deepcopy(config)
    trainer = QMixTrainer(seed_config, seed=seed)
    try:
        trainer.train()
        seed_wall_time = time.perf_counter() - seed_start_time
        return {
            "seed": seed,
            "env_steps": trainer.env_steps,
            "episodes": trainer.episode_count,
            "train_updates": trainer.train_updates,
            "seed_wall": seed_wall_time,
        }
    finally:
        trainer.close()
        del trainer
        release_training_memory()


def train(config: Optional[TrainConfig] = None):
    overall_start_time = time.perf_counter()
    config = build_run_config() if config is None else copy.deepcopy(config)
    normalize_config(config)

    resume_plan = None
    if config.run_mode == "fresh_start":
        seeds = resolve_seeds(config)
        if config.resume_checkpoint and len(seeds) != 1:
            raise ValueError("resume_checkpoint currently supports exactly one seed run.")
        manifest_path = write_seed_manifest(config, seeds)
    else:
        resume_plan = resolve_resume_plan(config)
        seeds = tuple(state["seed"] for state in resume_plan["pending_states"])
        manifest_path = resume_plan["manifest_path"]

    print(f"Seed manifest: {manifest_path}")

    resume_checkpoints = {}
    first_resume_checkpoint = None
    if resume_plan is not None:
        resume_checkpoints = {
            state["seed"]: str(state["checkpoint_path"]) if state["checkpoint_path"] else ""
            for state in resume_plan["pending_states"]
        }
        first_resume_checkpoint = resume_plan["pending_states"][0]["checkpoint_path"]
        completed_seeds = tuple(
            state["seed"] for state in resume_plan["seed_states"] if state["has_final"]
        )
        if completed_seeds:
            print(f"Resume skip | completed_seeds={completed_seeds}")
    elif config.resume_checkpoint and len(seeds) == 1:
        resume_checkpoints[seeds[0]] = config.resume_checkpoint
        first_resume_checkpoint = config.resume_checkpoint

    print(
        "Run plan | "
        f"mode={config.run_mode} | "
        f"num_seeds={len(seeds)} | "
        f"seeds={seeds} | "
        f"num_envs={config.num_envs} | "
        f"total_timesteps={config.total_timesteps} | "
        f"save_interval={config.save_interval} | "
        f"eval_during_training={config.eval_during_training} | "
        f"resume_checkpoint={first_resume_checkpoint if first_resume_checkpoint else 'None'}"
    )
    if resume_plan is not None:
        for state in resume_plan["pending_states"]:
            if state["save_dir"].is_dir() and state["checkpoint_path"] is None:
                print(
                    "Resume note | "
                    f"seed={state['seed']} | "
                    "unfinished seed folder has no step checkpoint yet, so training will restart from 0."
                )

    seed_results = []
    for seed in seeds:
        seed_config = copy.deepcopy(config)
        seed_config.resume_checkpoint = resume_checkpoints.get(seed, "")
        seed_results.append(train_for_seed(seed_config, seed))

    overall_wall_time = time.perf_counter() - overall_start_time
    if seed_results:
        print("Per-seed recap")
        for result in seed_results:
            print(
                "  - "
                f"seed={result['seed']} | "
                f"env_steps={result['env_steps']} | "
                f"episodes={result['episodes']} | "
                f"updates={result['train_updates']} | "
                f"seed_wall={result['seed_wall']:.2f}s"
            )
        total_steps = sum(result["env_steps"] for result in seed_results)
        total_seed_wall = sum(result["seed_wall"] for result in seed_results)
        print(
            "Overall run summary | "
            f"num_seeds={len(seed_results)} | "
            f"total_env_steps={total_steps} | "
            f"total_seed_wall={total_seed_wall:.2f}s | "
            f"overall_wall={overall_wall_time:.2f}s"
        )


def main():
    train(build_run_config())


if __name__ == "__main__":
    mp.freeze_support()
    main()
