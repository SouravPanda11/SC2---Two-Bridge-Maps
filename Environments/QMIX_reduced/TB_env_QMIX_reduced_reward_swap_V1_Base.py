from Environments.QMIX_reduced._qmix_reduced_reward_swap_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixRewardSwapReducedEnvBase,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V1_Base",
    registry_name="TwoBridgeMap_V1_Base",
    filename="TwoBridgeMap_V1_Base.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=3,
)


class TwoBridgeQMixRewardSwapEnv(TwoBridgeQMixRewardSwapReducedEnvBase):
    def __init__(self, map_name="V1_Base", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


class TwoBridgeQMixRewardSwapPathableOnlyEnv(TwoBridgeQMixRewardSwapEnv):
    def __init__(self, map_name="V1_Base", **kwargs):
        super().__init__(
            map_name=map_name,
            include_player_relative=False,
            **kwargs,
        )


TwoBridgeEnv = TwoBridgeQMixRewardSwapEnv
