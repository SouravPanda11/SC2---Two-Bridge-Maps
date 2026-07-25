from Environments.MAPPO_reduced._mappo_reduced_equal_terminal_reward_base import (
    TwoBridgeMAPPOEqualTerminalRewardReducedEnvBase,
    TwoBridgeMapConfig,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V2_Navigate",
    registry_name="TwoBridgeMap_V2_Navigate",
    filename="TwoBridgeMap_V2_Navigate.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=5,
)


class TwoBridgeMAPPOEqualTerminalRewardEnv(
    TwoBridgeMAPPOEqualTerminalRewardReducedEnvBase
):
    def __init__(self, map_name="V2_Navigate", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


class TwoBridgeMAPPOEqualTerminalRewardPathableOnlyEnv(
    TwoBridgeMAPPOEqualTerminalRewardEnv
):
    def __init__(self, map_name="V2_Navigate", **kwargs):
        super().__init__(
            map_name=map_name,
            include_player_relative=False,
            **kwargs,
        )


TwoBridgeEnv = TwoBridgeMAPPOEqualTerminalRewardEnv
