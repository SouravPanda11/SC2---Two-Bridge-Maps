from Environments.MultiAgent._qmix_maskppo_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixMaskPPOEnvBase,
)


MAP_CONFIG = TwoBridgeMapConfig(
    alias="V2_Base",
    registry_name="TwoBridgeMap_V2_Base",
    filename="TwoBridgeMap_V2_Base.SC2Map",
    directory=r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free",
    n_enemies=5,
)


class TwoBridgeQMixEnv(TwoBridgeQMixMaskPPOEnvBase):
    def __init__(self, map_name="V2_Base", **kwargs):
        super().__init__(map_config=MAP_CONFIG, map_name=map_name, **kwargs)


TwoBridgeEnv = TwoBridgeQMixEnv
