from Environments.MultiAgent._qmix_maskppo_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixMaskPPOEnvBase,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V1_Combat",
    registry_name="TwoBridgeMap_V1_Combat",
    filename="TwoBridgeMap_V1_Combat.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=3,
)


class TwoBridgeQMixEnv(TwoBridgeQMixMaskPPOEnvBase):
    def __init__(self, map_name="V1_Combat", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


TwoBridgeEnv = TwoBridgeQMixEnv
