from Environments.QMIX_reduced._qmix_reduced_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixReducedEnvBase,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V1_Combat",
    registry_name="TwoBridgeMap_V1_Combat",
    filename="TwoBridgeMap_V1_Combat.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=3,
)


class TwoBridgeQMixEnv(TwoBridgeQMixReducedEnvBase):
    def __init__(self, map_name="V1_Combat", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


class TwoBridgeQMixPathableOnlyEnv(TwoBridgeQMixEnv):
    def __init__(self, map_name="V1_Combat", **kwargs):
        super().__init__(map_name=map_name, include_player_relative=False, **kwargs)


TwoBridgeEnv = TwoBridgeQMixEnv

