"""
Isolated Reduced MaskPPO driver for the terminal-success-reward swap.

The established per-map trainer is reused without modifying it. Environment
factories passed to SubprocVecEnv capture the new reward-swap class explicitly,
so spawned Windows workers cannot fall back to the original environment.
"""

import atexit
import importlib
import os
import random
import re
import shutil
import tempfile
import time

import numpy as np


AGENT_NAME = "MaskPPO_NS_reduced_terminal_reward_swap"

TRAIN_MODULES = {
    "V1_Base": (
        "Agents.MaskPPO.V1_Base.MaskPPO_NS_reduced_parallel_train"
    ),
    "V2_Navigate": (
        "Agents.MaskPPO.V2_Navigate.MaskPPO_NS_reduced_parallel_train"
    ),
}

MAP_ENV_MODULES = {
    "V1_Base": (
        "Environments.NS_AM_RM_mean_reduced.V1_Base_reward_swap_NS"
    ),
    "V2_Navigate": (
        "Environments.NS_AM_RM_mean_reduced."
        "V2_Navigate_reward_swap_NS"
    ),
}


def _load_env_class(map_name: str):
    module = importlib.import_module(MAP_ENV_MODULES[map_name])
    return module.TwoBridgeEnv


def _make_spawn_safe_env_builder(base_module, env_class):
    flatten_wrapper = base_module.FlattenActionWrapper
    include_player_relative = True

    def make_env(rank, base_seed, visualize=False, realtime=False):
        def _init():
            worker_seed = base_seed + rank
            random.seed(worker_seed)
            np.random.seed(worker_seed)

            worker_tmp_dir = tempfile.mkdtemp(
                prefix=f"tbm-maskppo-reward-swap-worker-{rank}-"
            )
            atexit.register(
                lambda path=worker_tmp_dir: shutil.rmtree(
                    path,
                    ignore_errors=True,
                )
            )
            for key in ("TMP", "TEMP", "TMPDIR"):
                os.environ[key] = worker_tmp_dir

            time.sleep(0.5 * rank)
            env = env_class(
                visualize=visualize,
                realtime=realtime,
                include_player_relative=include_player_relative,
            )
            return flatten_wrapper(env)

        return _init

    return make_env


def train_with_settings(
    *,
    map_name: str,
    seed_values: tuple[int, ...],
    total_timesteps: int = 2_000_000,
    num_seeds: int = 2,
    num_envs: int = 3,
    save_interval: int = 50_000,
    run_mode: str = "fresh_start",
):
    if map_name not in TRAIN_MODULES:
        known = ", ".join(sorted(TRAIN_MODULES))
        raise ValueError(
            "Reward-swap Reduced MaskPPO supports only "
            f"{known}; received {map_name!r}."
        )
    if len(seed_values) != num_seeds:
        raise ValueError(
            "seed_values must contain exactly num_seeds entries; received "
            f"{len(seed_values)} values for num_seeds={num_seeds}."
        )

    base_module = importlib.import_module(TRAIN_MODULES[map_name])
    env_class = _load_env_class(map_name)
    make_env = _make_spawn_safe_env_builder(base_module, env_class)
    checkpoint_name_re = re.compile(
        rf"^{re.escape(AGENT_NAME)}_(\d+)([KMB]?)\.zip$",
        re.IGNORECASE,
    )

    def resolve_experiment_seeds(_args):
        return tuple(int(seed) for seed in seed_values)

    replacements = {
        "TwoBridgeEnv": env_class,
        "make_env": make_env,
        "resolve_seeds": resolve_experiment_seeds,
        "BASE_AGENT_NAME": AGENT_NAME,
        "AGENT_NAME": AGENT_NAME,
        "MAP_NAME": map_name,
        "CHECKPOINT_NAME_RE": checkpoint_name_re,
        "RUN_MODE": run_mode,
        "FRESH_START_SEED": None,
        "TOTAL_TIMESTEPS": int(total_timesteps),
        "SAVE_INTERVAL": int(save_interval),
        "NUM_SEEDS": int(num_seeds),
        "NUM_ENVS": int(num_envs),
        "INCLUDE_PLAYER_RELATIVE": True,
    }
    originals = {
        name: getattr(base_module, name)
        for name in replacements
    }

    for name, value in replacements.items():
        setattr(base_module, name, value)

    try:
        return base_module.main()
    finally:
        for name, value in originals.items():
            setattr(base_module, name, value)
