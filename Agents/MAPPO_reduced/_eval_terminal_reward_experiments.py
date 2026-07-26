"""
Evaluation adapter for the reduced MAPPO terminal-reward experiments.

It reuses the established checkpoint evaluation loop while routing spawned
workers to either the 10/25 reward-swap environments or the 25/25 equal-reward
environments. Existing evaluation and training modules are not modified.
"""

from __future__ import annotations

from pathlib import Path

from Agents import terminal_reward_eval_common as _common
from Agents.MAPPO_reduced import _eval_checkpoint_sweep as _base
from Agents.MAPPO_reduced import _train_mappo_reduced as _train_base
from Agents.MAPPO_reduced import (
    _train_mappo_reduced_equal_terminal_reward as _equal,
)
from Agents.MAPPO_reduced import (
    _train_mappo_reduced_reward_swap as _swap,
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


def _make_create_eval_env(env_modules):
    def create_eval_env(
        map_name,
        num_envs,
        seed,
        include_player_relative,
    ):
        if map_name not in env_modules:
            known = ", ".join(sorted(env_modules))
            raise ValueError(
                f"Terminal-reward MAPPO evaluation supports only {known}; "
                f"received {map_name!r}."
            )

        env_kwargs = {
            "map_name": map_name,
            "episode_limit": None,
            "visualize": False,
            "realtime": False,
            "replay_dir": "",
            "save_replay_episodes": 0,
            "include_player_relative": bool(include_player_relative),
        }
        original_modules = _train_base.MAP_ENV_MODULES
        _train_base.MAP_ENV_MODULES = dict(env_modules)
        try:
            return _base.ParallelEnvBatch(
                num_envs=int(num_envs),
                base_seed=int(seed),
                map_name=map_name,
                env_kwargs=env_kwargs,
            )
        finally:
            _train_base.MAP_ENV_MODULES = original_modules

    return create_eval_env


def _final_output_root(map_name, agent_name):
    return _common.final_evaluation_output_root(
        project_root=_base.PROJECT_ROOT,
        agent_directory="MAPPO_reduced",
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
            f"Terminal-reward MAPPO evaluation supports only {known}; "
            f"received {map_name!r}."
        )

    replacements = {
        "create_eval_env": _make_create_eval_env(env_modules),
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
