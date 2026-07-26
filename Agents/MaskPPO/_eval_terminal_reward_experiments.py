"""
Evaluation adapter for Reduced MaskPPO terminal-reward experiments.

The established checkpoint evaluator is reused. Spawned evaluation workers
capture the selected reward-condition module explicitly, preventing Windows
subprocesses from falling back to the original 25/10 environment.
"""

from __future__ import annotations

import atexit
import importlib
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from Agents import terminal_reward_eval_common as _common
from Agents.MaskPPO import _eval_ns_reduced_checkpoint_sweep as _base
from Agents.MaskPPO import (
    _train_maskppo_reduced_equal_terminal_reward as _equal,
)
from Agents.MaskPPO import (
    _train_maskppo_reduced_reward_swap as _swap,
)


EXPERIMENTS = {
    _common.EXPERIMENT_REWARD_SWAP: {
        "agent_name": _swap.AGENT_NAME,
        "env_modules": _swap.MAP_ENV_MODULES,
    },
    _common.EXPERIMENT_EQUAL_25: {
        "agent_name": _equal.AGENT_NAME,
        "env_modules": _equal.MAP_ENV_MODULES,
    },
}


def _make_condition_env(
    rank,
    base_seed,
    env_module_path,
    include_player_relative,
):
    def _init():
        worker_seed = int(base_seed) + int(rank)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        worker_tmp_dir = tempfile.mkdtemp(
            prefix=f"tbm-terminal-reward-eval-worker-{rank}-"
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
        env_module = importlib.import_module(env_module_path)
        env = env_module.TwoBridgeEnv(
            visualize=False,
            realtime=False,
            include_player_relative=bool(include_player_relative),
        )
        return _base.FlattenActionWrapper(
            env,
            n_friend=env_module.N_FRIEND,
        )

    return _init


def _make_create_vec_env(env_modules):
    def create_vec_env(
        num_envs,
        seed,
        map_name,
        include_player_relative,
    ):
        if map_name not in env_modules:
            known = ", ".join(sorted(env_modules))
            raise ValueError(
                "Terminal-reward Reduced MaskPPO evaluation supports only "
                f"{known}; received {map_name!r}."
            )
        env_module_path = env_modules[map_name]
        env_fns = [
            _make_condition_env(
                rank=rank,
                base_seed=seed,
                env_module_path=env_module_path,
                include_player_relative=include_player_relative,
            )
            for rank in range(int(num_envs))
        ]
        return _base.SubprocVecEnv(env_fns, start_method="spawn")

    return create_vec_env


def _final_output_root(map_name, agent_name):
    return _common.final_evaluation_output_root(
        project_root=_base.PROJECT_ROOT,
        agent_directory="MaskPPO",
        map_name=map_name,
        agent_name=agent_name,
    )


def main_for_script(
    script_path,
    *,
    experiment,
    final_only=False,
):
    experiment = _common.validate_experiment(experiment)
    settings = EXPERIMENTS[experiment]
    map_name = Path(script_path).resolve().parent.name
    env_modules = settings["env_modules"]
    if map_name not in env_modules:
        known = ", ".join(sorted(env_modules))
        raise ValueError(
            "Terminal-reward Reduced MaskPPO evaluation supports only "
            f"{known}; received {map_name!r}."
        )

    replacements = {
        "create_vec_env": _make_create_vec_env(env_modules),
        "env_module_name": lambda selected_map: env_modules[selected_map],
    }
    if final_only:
        replacements.update(
            {
                "collect_seed_checkpoints": (
                    lambda save_root, agent_name, seed:
                    _common.collect_final_checkpoint(
                        save_root,
                        agent_name,
                        seed,
                        ".zip",
                    )
                ),
                "get_output_root": _final_output_root,
            }
        )

    originals = {
        name: getattr(_base, name)
        for name in replacements
    }
    for name, value in replacements.items():
        setattr(_base, name, value)

    try:
        return _base.main(
            default_map_name=map_name,
            default_agent_name=settings["agent_name"],
        )
    finally:
        for name, value in originals.items():
            setattr(_base, name, value)
