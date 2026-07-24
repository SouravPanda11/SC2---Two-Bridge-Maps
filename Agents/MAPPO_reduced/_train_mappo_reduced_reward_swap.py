"""
Isolated MAPPO driver for the terminal-success-reward swap experiment.

The established MAPPO implementation is reused without modifying it. During
this call its environment lookup is scoped to the two reward-swap modules and
restored when training finishes.
"""

from Agents.MAPPO_reduced import _train_mappo_reduced as _base


AGENT_NAME = "MAPPO_reduced_terminal_reward_swap"

MAP_ENV_MODULES = {
    "V1_Base": (
        "Environments.MAPPO_reduced."
        "TB_env_MAPPO_reduced_reward_swap_V1_Base"
    ),
    "V2_Navigate": (
        "Environments.MAPPO_reduced."
        "TB_env_MAPPO_reduced_reward_swap_V2_Navigate"
    ),
}


def train_with_settings(map_name: str, **overrides):
    if map_name not in MAP_ENV_MODULES:
        known = ", ".join(sorted(MAP_ENV_MODULES))
        raise ValueError(
            f"Reward-swap MAPPO supports only {known}; received {map_name!r}."
        )

    requested_agent_name = overrides.setdefault("agent_name", AGENT_NAME)
    if not requested_agent_name.startswith(AGENT_NAME):
        raise ValueError(
            "Reward-swap outputs must use the isolated agent-name prefix "
            f"{AGENT_NAME!r}."
        )

    original_modules = _base.MAP_ENV_MODULES
    _base.MAP_ENV_MODULES = dict(MAP_ENV_MODULES)
    try:
        return _base.train_with_settings(map_name=map_name, **overrides)
    finally:
        _base.MAP_ENV_MODULES = original_modules
