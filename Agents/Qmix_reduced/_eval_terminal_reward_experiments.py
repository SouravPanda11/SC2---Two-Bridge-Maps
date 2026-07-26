"""
Evaluation adapter for the reduced QMIX terminal-reward experiments.

The established QMIX checkpoint evaluator and policy reconstruction are
reused. A lightweight trainer proxy replaces only the parallel environment
batch so Windows workers load the intended reward-condition environment.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from Agents import terminal_reward_eval_common as _common
from Agents.Qmix_reduced import _eval_checkpoint_sweep as _base
from Agents.Qmix_reduced import (
    _train_qmix_reduced_equal_terminal_reward as _equal,
)
from Agents.Qmix_reduced import (
    _train_qmix_reduced_reward_swap as _swap,
)


EXPERIMENTS = {
    _common.EXPERIMENT_REWARD_SWAP: {
        "agent_name": _swap.AGENT_NAME,
        "driver": _swap,
    },
    _common.EXPERIMENT_EQUAL_25: {
        "agent_name": _equal.AGENT_NAME,
        "driver": _equal,
    },
}


class _TrainerProxy:
    def __init__(self, base_module, parallel_env_batch, map_name):
        self._base_module = base_module
        self.ParallelQMixEnvBatch = parallel_env_batch
        self.MAP_NAME = map_name

    def __getattr__(self, name):
        return getattr(self._base_module, name)


def _make_load_trainer_module(experiment_driver):
    def load_trainer_module(map_name):
        if map_name not in experiment_driver.TRAIN_MODULES:
            known = ", ".join(sorted(experiment_driver.TRAIN_MODULES))
            raise ValueError(
                f"Terminal-reward QMIX evaluation supports only {known}; "
                f"received {map_name!r}."
            )

        base_module = importlib.import_module(
            experiment_driver.TRAIN_MODULES[map_name]
        )
        parallel_env_batch = experiment_driver._make_parallel_env_batch(
            base_module
        )
        return _TrainerProxy(
            base_module=base_module,
            parallel_env_batch=parallel_env_batch,
            map_name=map_name,
        )

    return load_trainer_module


def _final_output_root(map_name, agent_name):
    return _common.final_evaluation_output_root(
        project_root=_base.PROJECT_ROOT,
        agent_directory="Qmix_reduced",
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
    experiment_driver = settings["driver"]
    map_name = Path(script_path).resolve().parent.name
    if map_name not in experiment_driver.MAP_ENV_MODULES:
        known = ", ".join(sorted(experiment_driver.MAP_ENV_MODULES))
        raise ValueError(
            f"Terminal-reward QMIX evaluation supports only {known}; "
            f"received {map_name!r}."
        )

    replacements = {
        "load_trainer_module": _make_load_trainer_module(
            experiment_driver
        ),
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
                        ".pt",
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
