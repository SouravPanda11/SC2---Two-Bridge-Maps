import argparse
import atexit
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import time

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
AGENT_NAME = "MaskPPO_NS_AM_RM_mean_parallel"
MAP_NAME = "V2_Base"


class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) ->
    MultiDiscrete([3, 2xN_FRIEND, 9, N_ENEMY+1])
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete([3] + [2] * N_FRIEND + [9] + [N_ENEMY + 1])

        # Bits beyond the verb-level mask that are always legal.
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        # Advertise the flattened mask directly on the wrapped env so
        # MaskablePPO can query it from SubprocVecEnv workers.
        flat_len = 3 + len(self._mask_template)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)
        self._last_mask = np.ones(flat_len, dtype=np.int8)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb": int(a_vec[0]),
            "who": np.asarray(a_vec[1 : 1 + N_FRIEND], np.int8),
            "direction": int(a_vec[1 + N_FRIEND]),
            "enemy_idx": int(a_vec[-1]),
        }

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        flat_mask = np.concatenate([obs["action_mask"], self._mask_template]).astype(np.int8)
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


def get_output_dirs(seed):
    save_dir = os.path.join(
        project_root,
        "Agents",
        "MaskPPO",
        MAP_NAME,
        "saved_models",
        AGENT_NAME,
        f"seed_{seed}",
    )
    tb_log_dir = os.path.join(
        project_root,
        "tb_logs",
        "MaskPPO",
        MAP_NAME,
        AGENT_NAME,
        f"seed_{seed}",
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    return save_dir, tb_log_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on V1 Base NS with SubprocVecEnv workers."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Train a single explicit seed. If omitted, random seeds are generated.",
    )
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny training job to validate multiprocessing setup.",
    )
    return parser.parse_args()


def normalize_args(args):
    if args.smoke_test:
        args.total_timesteps = max(args.num_envs * 16, 32)
        args.save_interval = args.total_timesteps
        args.n_steps = 16
        args.batch_size = args.num_envs * args.n_steps
        args.n_epochs = 1

    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")
    if args.total_timesteps < 1:
        raise ValueError("--total-timesteps must be at least 1")
    if args.save_interval < 1:
        raise ValueError("--save-interval must be at least 1")
    if args.n_steps < 2:
        raise ValueError("--n-steps must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be at least 1")
    if args.save_interval > args.total_timesteps:
        raise ValueError("--save-interval cannot exceed --total-timesteps")

    rollout_size = args.num_envs * args.n_steps
    if args.batch_size > rollout_size:
        raise ValueError(
            f"--batch-size ({args.batch_size}) cannot exceed rollout size ({rollout_size})"
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


def train_for_seed(args, rollout_size, seed):
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
    args = parse_args()
    rollout_size = normalize_args(args)
    seeds = resolve_seeds(args)

    print(
        "Multi-seed plan | "
        f"num_seeds={len(seeds)} | "
        f"seeds={seeds} | "
        f"num_envs={args.num_envs} | "
        f"total_timesteps={args.total_timesteps} | "
        f"save_interval={args.save_interval}"
    )

    seed_results = []
    for seed in seeds:
        seed_results.append(train_for_seed(args, rollout_size, seed))

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
