from Environments.MAPPO_reduced._mappo_reduced_base import (
    TwoBridgeMAPPOReducedEnvBase,
    TwoBridgeMapConfig,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V3_Combat",
    registry_name="TwoBridgeMap_V3_Combat",
    filename="TwoBridgeMap_V3_Combat.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=8,
)


class TwoBridgeMAPPOEnv(TwoBridgeMAPPOReducedEnvBase):
    def __init__(self, map_name="V3_Combat", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


class TwoBridgeMAPPOPathableOnlyEnv(TwoBridgeMAPPOEnv):
    def __init__(self, map_name="V3_Combat", **kwargs):
        super().__init__(map_name=map_name, include_player_relative=False, **kwargs)


TwoBridgeEnv = TwoBridgeMAPPOEnv
