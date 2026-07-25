from Environments.MAPPO_reduced._mappo_reduced_equal_terminal_reward_base import (
    TwoBridgeMAPPOEqualTerminalRewardReducedEnvBase,
    TwoBridgeMapConfig,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V1_Base",
    registry_name="TwoBridgeMap_V1_Base",
    filename="TwoBridgeMap_V1_Base.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=3,
)


class TwoBridgeMAPPOEqualTerminalRewardEnv(
    TwoBridgeMAPPOEqualTerminalRewardReducedEnvBase
):
    def __init__(self, map_name="V1_Base", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


class TwoBridgeMAPPOEqualTerminalRewardPathableOnlyEnv(
    TwoBridgeMAPPOEqualTerminalRewardEnv
):
    def __init__(self, map_name="V1_Base", **kwargs):
        super().__init__(
            map_name=map_name,
            include_player_relative=False,
            **kwargs,
        )


TwoBridgeEnv = TwoBridgeMAPPOEqualTerminalRewardEnv
