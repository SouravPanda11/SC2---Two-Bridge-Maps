from Environments.MultiAgent._qmix_maskppo_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixMaskPPOEnvBase,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V3_Combat",
    registry_name="TwoBridgeMap_V3_Combat",
    filename="TwoBridgeMap_V3_Combat.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=8,
)


class TwoBridgeQMixEnv(TwoBridgeQMixMaskPPOEnvBase):
    def __init__(self, map_name="V3_Combat", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


TwoBridgeEnv = TwoBridgeQMixEnv
