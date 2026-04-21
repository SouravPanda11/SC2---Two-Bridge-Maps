import atexit
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from Environments.NS_AM_RM_mean.V2_Base_NS import TwoBridgeEnv, N_FRIEND, N_ENEMY


DEFAULT_TOTAL_TIMESTEPS = 2_000_000
DEFAULT_SAVE_INTERVAL = 50_000
DEFAULT_NUM_ENVS = 3
DEFAULT_NUM_SEEDS = 3
DEFAULT_N_STEPS = 512
DEFAULT_BATCH_SIZE = 512
DEFAULT_N_EPOCHS = 4

# Change the env import above plus these names to reuse the trainer for
# other map variants without touching the rollout logic below.
AGENT_NAME = "MaskPPO_NS"
MAP_NAME = "V2_Base"
SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
CHECKPOINT_NAME_RE = re.compile(
    rf"^{re.escape(AGENT_NAME)}_(\d+)([KMB]?)\.zip$",
    re.IGNORECASE,
)

# ============================================================================
# Run mode: comment/uncomment exactly one option below.
# ============================================================================
RUN_MODE = "fresh_start"
# RUN_MODE = "load_last_checkpoint"

# Fresh start settings.
FRESH_START_SEED = None  # Set an int to train one explicit seed from scratch.

# Training settings.
TOTAL_TIMESTEPS = DEFAULT_TOTAL_TIMESTEPS
SAVE_INTERVAL = DEFAULT_SAVE_INTERVAL
NUM_ENVS = DEFAULT_NUM_ENVS
NUM_SEEDS = DEFAULT_NUM_SEEDS
N_STEPS = DEFAULT_N_STEPS
BATCH_SIZE = DEFAULT_BATCH_SIZE
N_EPOCHS = DEFAULT_N_EPOCHS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISUALIZE = False
REALTIME = False
SMOKE_TEST = False


class FlattenActionWrapper(Wrapper):
    """
    Keeps the NS env's per-friendly MultiDiscrete action space and flattens
    the 2D action mask so MaskablePPO can consume it.
    """

    def __init__(self, env):
        super().__init__(env)

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise TypeError("Expected MultiDiscrete action_space from NS env")
        self.action_space = env.action_space
        self._n_agents = int(len(self.action_space.nvec))
        if self._n_agents != N_FRIEND:
            raise ValueError(
                f"Unexpected action_space rank {self._n_agents}; expected N_FRIEND={N_FRIEND}"
            )
        if np.unique(self.action_space.nvec).size != 1:
            raise ValueError("Expected identical per-friendly action dimensions")
        self._n_unit_actions = int(self.action_space.nvec[0])

        # Advertise the flattened mask directly on the wrapped env so
        # MaskablePPO can query it from SubprocVecEnv workers.
        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)
        self._last_mask = np.ones(flat_len, dtype=np.int8)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        obs, rew, term, trunc, info = self.env.step(action_arr)
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        am = np.asarray(obs["action_mask"], dtype=np.int8)
        expected_shape = (self._n_agents, self._n_unit_actions)
        if am.shape != expected_shape:
            raise ValueError(
                f"action_mask shape {am.shape} != expected {expected_shape}"
            )

        flat_mask = am.reshape(-1).astype(np.int8, copy=True)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask


class TBRewardLogger(BaseCallback):
    """
    Logs mean env-provided reward components under 'rew/*' in TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos or self.logger is None:
            return True

        totals = {}
        counts = {}
        for info in infos:
            if not isinstance(info, dict) or "rew" not in info:
                continue
            for key, value in info["rew"].items():
                try:
                    val = float(value)
                except Exception:
                    continue
                totals[key] = totals.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1

        for key, total in totals.items():
            self.logger.record(f"rew/{key}", total / counts[key])
        return True


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_single_env(seed):
    base_env = TwoBridgeEnv(visualize=False, realtime=False)
    flat_env = FlattenActionWrapper(base_env)
    try:
        obs, _ = flat_env.reset(seed=seed)
        expected_keys = {"minimap", "vector", "action_mask"}
        if set(obs.keys()) != expected_keys:
            raise RuntimeError(f"Unexpected observation keys: {sorted(obs.keys())}")

        expected_minimap_shape = flat_env.observation_space["minimap"].shape
        expected_vector_shape = flat_env.observation_space["vector"].shape
        expected_mask_shape = flat_env.observation_space["action_mask"].shape

        if obs["minimap"].shape != expected_minimap_shape:
            raise RuntimeError(
                f"Unexpected minimap shape: {obs['minimap'].shape} != {expected_minimap_shape}"
            )
        if obs["vector"].shape != expected_vector_shape:
            raise RuntimeError(
                f"Unexpected vector shape: {obs['vector'].shape} != {expected_vector_shape}"
            )
        if obs["action_mask"].shape != expected_mask_shape:
            raise RuntimeError(
                f"Unexpected action_mask shape: {obs['action_mask'].shape} != {expected_mask_shape}"
            )

        print(
            "Obs contract OK | "
            f"seed={seed} | "
            f"keys={sorted(obs.keys())} | "
            f"minimap={obs['minimap'].shape} | "
            f"vector={obs['vector'].shape} | "
            f"action_mask={obs['action_mask'].shape}"
        )
    finally:
        flat_env.close()


def make_env(rank, base_seed, visualize=False, realtime=False):
    def _init():
        worker_seed = base_seed + rank
        random.seed(worker_seed)
        np.random.seed(worker_seed)

        # Give each SC2 worker a private temp root so the engine does not
        # race on TempLaunchMap.SC2Map during parallel startup on Windows.
        worker_tmp_dir = tempfile.mkdtemp(prefix=f"tbm-sc2-worker-{rank}-")
        atexit.register(lambda path=worker_tmp_dir: shutil.rmtree(path, ignore_errors=True))
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = worker_tmp_dir

        # A short stagger reduces startup collisions while ports and temp
        # files are still being allocated.
        time.sleep(0.5 * rank)

        env = TwoBridgeEnv(visualize=visualize, realtime=realtime)
        env = FlattenActionWrapper(env)
        return env

    return _init


def create_vec_env(num_envs, seed, visualize=False, realtime=False):
    env_fns = [
        make_env(rank=rank, base_seed=seed, visualize=visualize, realtime=realtime)
        for rank in range(num_envs)
    ]
    return SubprocVecEnv(env_fns, start_method="spawn")


def format_step_label(total_steps):
    if total_steps % 1000 == 0:
        return f"{total_steps // 1000}K"
    return str(total_steps)


def get_agent_save_root():
    return os.path.join(project_root, "Agents", "MaskPPO", MAP_NAME, "saved_models", AGENT_NAME)


def get_agent_tb_root():
    return os.path.join(project_root, "tb_logs", "MaskPPO", MAP_NAME, AGENT_NAME)


def get_output_dirs(seed):
    save_dir = os.path.join(get_agent_save_root(), f"seed_{seed}")
    tb_log_dir = os.path.join(get_agent_tb_root(), f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    return save_dir, tb_log_dir


def build_run_config():
    return SimpleNamespace(
        run_mode=RUN_MODE,
        seed=FRESH_START_SEED,
        num_seeds=NUM_SEEDS,
        num_envs=NUM_ENVS,
        total_timesteps=TOTAL_TIMESTEPS,
        save_interval=SAVE_INTERVAL,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        visualize=VISUALIZE,
        realtime=REALTIME,
        smoke_test=SMOKE_TEST,
    )


def write_seed_manifest(config, seeds):
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent_name": AGENT_NAME,
        "map_name": MAP_NAME,
        "run_mode": config.run_mode,
        "seeds": list(seeds),
        "num_envs": config.num_envs,
        "total_timesteps": config.total_timesteps,
        "save_interval": config.save_interval,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "device": config.device,
    }
    save_root = get_agent_save_root()
    os.makedirs(save_root, exist_ok=True)

    manifest_name = f"run_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    latest_manifest_path = os.path.join(save_root, "latest_run_manifest.json")
    dated_manifest_path = os.path.join(save_root, manifest_name)
    for path in (latest_manifest_path, dated_manifest_path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    return latest_manifest_path


def iter_seed_dirs():
    save_root = get_agent_save_root()
    if not os.path.isdir(save_root):
        return

    for entry in os.scandir(save_root):
        if not entry.is_dir():
            continue
        match = SEED_DIR_RE.fullmatch(entry.name)
        if match is None:
            continue
        yield int(match.group(1)), entry.path


def is_final_checkpoint_name(checkpoint_name):
    return checkpoint_name.endswith(".zip") and "_final" in os.path.splitext(checkpoint_name)[0]


def parse_checkpoint_steps(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    if is_final_checkpoint_name(checkpoint_name):
        return None

    match = CHECKPOINT_NAME_RE.fullmatch(checkpoint_name)
    if match is None:
        return None

    value = int(match.group(1))
    suffix = match.group(2).upper()
    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers[suffix]


def checkpoint_sort_key(checkpoint_path):
    parsed_steps = parse_checkpoint_steps(checkpoint_path)
    if parsed_steps is None:
        return None
    return (parsed_steps, os.path.getmtime(checkpoint_path))


def collect_seed_checkpoints(seed):
    save_dir = os.path.join(get_agent_save_root(), f"seed_{seed}")
    if not os.path.isdir(save_dir):
        return []

    checkpoint_paths = []
    for entry in os.scandir(save_dir):
        if not entry.is_file() or not entry.name.endswith(".zip"):
            continue
        if is_final_checkpoint_name(entry.name):
            continue
        if checkpoint_sort_key(entry.path) is None:
            continue
        checkpoint_paths.append(entry.path)
    return checkpoint_paths


def seed_dir_has_final(seed_dir):
    for entry in os.scandir(seed_dir):
        if entry.is_file() and is_final_checkpoint_name(entry.name):
            return True
    return False


def load_latest_seed_manifest():
    manifest_path = os.path.join(get_agent_save_root(), "latest_run_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No seed manifest found at {manifest_path}. Run fresh_start first to create it."
        )

    with open(manifest_path, "r", encoding="utf-8") as handle:
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
            raise RuntimeError(f"Seed manifest {manifest_path} contains duplicate seed {seed}.")
        seeds.append(seed)
        seen.add(seed)

    manifest["seeds"] = tuple(seeds)
    return manifest_path, manifest


def describe_seed_progress(seed):
    save_dir = os.path.join(get_agent_save_root(), f"seed_{seed}")
    checkpoint_paths = collect_seed_checkpoints(seed)
    checkpoint_path = max(checkpoint_paths, key=checkpoint_sort_key) if checkpoint_paths else None
    return {
        "seed": seed,
        "save_dir": save_dir,
        "has_final": os.path.isdir(save_dir) and seed_dir_has_final(save_dir),
        "checkpoint_path": checkpoint_path,
    }


def resolve_resume_plan():
    manifest_path, manifest = load_latest_seed_manifest()
    seed_states = [describe_seed_progress(seed) for seed in manifest["seeds"]]
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


def normalize_config(config):
    if config.run_mode not in {"fresh_start", "load_last_checkpoint"}:
        raise ValueError(
            f"Invalid RUN_MODE: {config.run_mode!r}. Use 'fresh_start' or 'load_last_checkpoint'."
        )

    if config.smoke_test:
        config.total_timesteps = max(config.num_envs * 16, 32)
        config.save_interval = config.total_timesteps
        config.n_steps = 16
        config.batch_size = config.num_envs * config.n_steps
        config.n_epochs = 1

    if config.num_envs < 1:
        raise ValueError("NUM_ENVS must be at least 1")
    if config.num_seeds < 1:
        raise ValueError("NUM_SEEDS must be at least 1")
    if config.total_timesteps < 1:
        raise ValueError("TOTAL_TIMESTEPS must be at least 1")
    if config.save_interval < 1:
        raise ValueError("SAVE_INTERVAL must be at least 1")
    if config.n_steps < 2:
        raise ValueError("N_STEPS must be at least 2")
    if config.batch_size < 1:
        raise ValueError("BATCH_SIZE must be at least 1")
    if config.n_epochs < 1:
        raise ValueError("N_EPOCHS must be at least 1")
    if config.save_interval > config.total_timesteps:
        raise ValueError("SAVE_INTERVAL cannot exceed TOTAL_TIMESTEPS")

    rollout_size = config.num_envs * config.n_steps
    if config.batch_size > rollout_size:
        raise ValueError(
            f"BATCH_SIZE ({config.batch_size}) cannot exceed rollout size ({rollout_size})"
        )
    return rollout_size


def build_checkpoint_targets(total_timesteps, save_interval):
    targets = list(range(save_interval, total_timesteps + 1, save_interval))
    if not targets or targets[-1] != total_timesteps:
        targets.append(total_timesteps)
    return targets


def generate_random_seeds(num_seeds):
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


def resolve_seeds(args):
    if args.seed is not None:
        return (args.seed,)
    if args.smoke_test:
        return (0,)
    return generate_random_seeds(args.num_seeds)


def train_for_seed(args, rollout_size, seed, resume_checkpoint=None):
    seed_start_time = time.perf_counter()
    set_global_seeds(seed)

    validate_start_time = time.perf_counter()
    validate_single_env(seed)
    validate_wall_time = time.perf_counter() - validate_start_time

    save_dir, tb_log_dir = get_output_dirs(seed)
    print(
        f"Starting seed run | "
        f"seed={seed} | "
        f"Using device: {args.device} | "
        f"num_envs={args.num_envs} | "
        f"n_steps={args.n_steps} | "
        f"batch_size={args.batch_size} | "
        f"n_epochs={args.n_epochs} | "
        f"rollout_size={rollout_size} | "
        f"total_timesteps={args.total_timesteps} | "
        f"save_interval={args.save_interval}"
    )
    print(
        "Timing note | "
        "PPO collects full rollouts of num_envs * n_steps, "
        "so actual checkpoint steps land on the first rollout boundary at or after each target."
    )
    print(f"Single-env validation wall time: {validate_wall_time:.2f}s")
    print(f"Checkpoint dir: {save_dir}")
    print(f"TensorBoard dir: {tb_log_dir}")
    if resume_checkpoint is not None:
        print(f"Resume checkpoint: {resume_checkpoint}")

    env_create_start_time = time.perf_counter()
    env = create_vec_env(
        num_envs=args.num_envs,
        seed=seed,
        visualize=args.visualize,
        realtime=args.realtime,
    )
    env_create_wall_time = time.perf_counter() - env_create_start_time
    try:
        env.seed(seed)
        env_reset_start_time = time.perf_counter()
        obs = env.reset()
        masks = env.env_method("action_masks")
        env_reset_wall_time = time.perf_counter() - env_reset_start_time

        print(
            "Vector env reset OK | "
            f"minimap={obs['minimap'].shape} | "
            f"vector={obs['vector'].shape} | "
            f"action_mask={obs['action_mask'].shape} | "
            f"mask_workers={len(masks)} | "
            f"mask_shape={np.asarray(masks[0]).shape}"
        )
        print(
            "Setup timing | "
            f"vec_env_create={env_create_wall_time:.2f}s | "
            f"first_reset={env_reset_wall_time:.2f}s"
        )

        if resume_checkpoint is None:
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                device=args.device,
                verbose=1,
                tensorboard_log=tb_log_dir,
                seed=seed,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
            )
        else:
            model = MaskablePPO.load(
                resume_checkpoint,
                env=env,
                device=args.device,
                tensorboard_log=tb_log_dir,
                seed=seed,
            )
            print(
                "Resume state loaded | "
                f"seed={seed} | "
                f"checkpoint_steps={model.num_timesteps} | "
                f"remaining_steps={max(args.total_timesteps - model.num_timesteps, 0)}"
            )

        tb_callback = TBRewardLogger()
        training_start_time = time.perf_counter()
        checkpoint_targets = build_checkpoint_targets(args.total_timesteps, args.save_interval)
        total_requested_steps = 0

        for chunk_index, target_step in enumerate(checkpoint_targets, start=1):
            chunk_requested_steps = max(target_step - model.num_timesteps, 0)
            if chunk_requested_steps == 0:
                continue
            chunk_start_timesteps = model.num_timesteps
            chunk_learn_start_time = time.perf_counter()
            model.learn(
                total_timesteps=chunk_requested_steps,
                reset_num_timesteps=False,
                callback=tb_callback,
                progress_bar=True,
                tb_log_name=f"nenv_{args.num_envs}",
            )
            chunk_learn_wall_time = time.perf_counter() - chunk_learn_start_time
            chunk_actual_steps = model.num_timesteps - chunk_start_timesteps

            save_start_time = time.perf_counter()
            total_requested_steps += chunk_requested_steps
            checkpoint_name = f"{AGENT_NAME}_{format_step_label(model.num_timesteps)}"
            model.save(os.path.join(save_dir, checkpoint_name))
            save_wall_time = time.perf_counter() - save_start_time
            chunk_total_wall_time = chunk_learn_wall_time + save_wall_time
            transitions_per_sec = (
                chunk_actual_steps / chunk_learn_wall_time if chunk_learn_wall_time > 0 else float("inf")
            )
            vec_steps = chunk_actual_steps / args.num_envs
            vec_steps_per_sec = (
                vec_steps / chunk_learn_wall_time if chunk_learn_wall_time > 0 else float("inf")
            )

            print(
                "Timing checkpoint | "
                f"index={chunk_index} | "
                f"seed={seed} | "
                f"target_steps={target_step} | "
                f"requested_steps={chunk_requested_steps} | "
                f"actual_chunk_steps={chunk_actual_steps} | "
                f"actual_total_steps={model.num_timesteps} | "
                f"learn_wall={chunk_learn_wall_time:.2f}s | "
                f"save_wall={save_wall_time:.2f}s | "
                f"chunk_wall={chunk_total_wall_time:.2f}s | "
                f"transitions_per_sec={transitions_per_sec:.2f} | "
                f"vec_steps_per_sec={vec_steps_per_sec:.2f}"
            )
            print(
                "Saved checkpoint | "
                f"seed={seed} | "
                f"target_steps={target_step} | "
                f"actual_total_steps={model.num_timesteps}"
            )

        model.save(os.path.join(save_dir, f"{AGENT_NAME}_final"))
        training_wall_time = time.perf_counter() - training_start_time
        seed_wall_time = time.perf_counter() - seed_start_time
        effective_total_steps = model.num_timesteps
        effective_transitions_per_sec = (
            effective_total_steps / training_wall_time if training_wall_time > 0 else float("inf")
        )
        effective_vec_steps_per_sec = (
            (effective_total_steps / args.num_envs) / training_wall_time
            if training_wall_time > 0
            else float("inf")
        )

        print(
            "Training summary | "
            f"seed={seed} | "
            f"num_envs={args.num_envs} | "
            f"requested_total_steps={total_requested_steps} | "
            f"effective_total_steps={effective_total_steps} | "
            f"training_wall={training_wall_time:.2f}s | "
            f"seed_wall={seed_wall_time:.2f}s | "
            f"transitions_per_sec={effective_transitions_per_sec:.2f} | "
            f"vec_steps_per_sec={effective_vec_steps_per_sec:.2f}"
        )
        print(
            "Finished training | "
            f"seed={seed} | "
            f"requested_total_steps={args.total_timesteps} | "
            f"actual_total_steps={model.num_timesteps}"
        )
        return {
            "seed": seed,
            "requested_total_steps": total_requested_steps,
            "effective_total_steps": effective_total_steps,
            "training_wall": training_wall_time,
            "seed_wall": seed_wall_time,
            "transitions_per_sec": effective_transitions_per_sec,
            "vec_steps_per_sec": effective_vec_steps_per_sec,
        }
    finally:
        env.close()


def main():
    overall_start_time = time.perf_counter()
    args = build_run_config()
    rollout_size = normalize_config(args)
    resume_plan = None

    if args.run_mode == "fresh_start":
        seeds = resolve_seeds(args)
        manifest_path = write_seed_manifest(args, seeds)
        print(f"Seed manifest: {manifest_path}")
    else:
        resume_plan = resolve_resume_plan()
        seeds = tuple(state["seed"] for state in resume_plan["pending_states"])
        print(f"Seed manifest: {resume_plan['manifest_path']}")

    resume_checkpoints = {}
    first_resume_checkpoint = None
    if resume_plan is not None:
        resume_checkpoints = {
            state["seed"]: state["checkpoint_path"] for state in resume_plan["pending_states"]
        }
        first_resume_checkpoint = resume_plan["pending_states"][0]["checkpoint_path"]
        completed_seeds = tuple(
            state["seed"] for state in resume_plan["seed_states"] if state["has_final"]
        )
        if completed_seeds:
            print(f"Resume skip | completed_seeds={completed_seeds}")

    print(
        "Run plan | "
        f"mode={args.run_mode} | "
        f"num_seeds={len(seeds)} | "
        f"seeds={seeds} | "
        f"num_envs={args.num_envs} | "
        f"total_timesteps={args.total_timesteps} | "
        f"save_interval={args.save_interval} | "
        f"resume_checkpoint={first_resume_checkpoint if first_resume_checkpoint else 'None'}"
    )
    if resume_plan is not None:
        for state in resume_plan["pending_states"]:
            if os.path.isdir(state["save_dir"]) and state["checkpoint_path"] is None:
                print(
                    "Resume note | "
                    f"seed={state['seed']} | "
                    "unfinished seed folder has no step checkpoint yet, so training will restart from 0."
                )

    seed_results = []
    for seed in seeds:
        seed_results.append(train_for_seed(args, rollout_size, seed, resume_checkpoints.get(seed)))

    overall_wall_time = time.perf_counter() - overall_start_time
    total_training_wall = sum(result["training_wall"] for result in seed_results)
    total_seed_wall = sum(result["seed_wall"] for result in seed_results)
    total_effective_steps = sum(result["effective_total_steps"] for result in seed_results)
    aggregate_transitions_per_sec = (
        total_effective_steps / total_training_wall if total_training_wall > 0 else float("inf")
    )

    print("Per-seed timing recap")
    for result in seed_results:
        print(
            "  - "
            f"seed={result['seed']} | "
            f"training_wall={result['training_wall']:.2f}s | "
            f"seed_wall={result['seed_wall']:.2f}s | "
            f"effective_total_steps={result['effective_total_steps']} | "
            f"transitions_per_sec={result['transitions_per_sec']:.2f}"
        )

    print(
        "Overall run summary | "
        f"num_seeds={len(seeds)} | "
        f"seeds={seeds} | "
        f"total_training_wall={total_training_wall:.2f}s | "
        f"total_seed_wall={total_seed_wall:.2f}s | "
        f"overall_wall={overall_wall_time:.2f}s | "
        f"total_effective_steps={total_effective_steps} | "
        f"aggregate_transitions_per_sec={aggregate_transitions_per_sec:.2f}"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
