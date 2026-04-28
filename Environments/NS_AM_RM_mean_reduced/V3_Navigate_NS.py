from __future__ import annotations

import numpy as np
from gymnasium import spaces

from Environments.NS_AM_RM_mean.V3_Navigate_NS import *  # noqa: F401,F403
from Environments.NS_AM_RM_mean.V3_Navigate_NS import TwoBridgeEnv as _FullTwoBridgeEnv
from Environments.NS_AM_RM_mean_reduced._reduced_minimap import (
    DEFAULT_MINIMAP_CROP,
    REDUCED_MINIMAP_SHAPE,
    reduce_minimap,
    reduced_minimap_shape,
)


class TwoBridgeEnv(_FullTwoBridgeEnv):
    minimap_crop = DEFAULT_MINIMAP_CROP

    observation_space = spaces.Dict({
        "minimap": spaces.Box(0, 4, REDUCED_MINIMAP_SHAPE, np.uint8),
        "vector": spaces.Box(0.0, np.inf, (OBS_VEC_SIZE,), np.float32),
        "action_mask": spaces.MultiBinary((N_FRIEND, N_UNIT_ACTIONS)),
    })

    def __init__(self, *args, include_player_relative: bool = True, **kwargs):
        self.include_player_relative = bool(include_player_relative)
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict({
            "minimap": spaces.Box(
                0, 4, reduced_minimap_shape(self.include_player_relative), np.uint8
            ),
            "vector": spaces.Box(0.0, np.inf, (OBS_VEC_SIZE,), np.float32),
            "action_mask": spaces.MultiBinary((N_FRIEND, N_UNIT_ACTIONS)),
        })

    def _build_obs(self, ts):
        obs = super()._build_obs(ts)
        obs["minimap"] = reduce_minimap(
            obs["minimap"], self.minimap_crop, self.include_player_relative
        )
        return obs


class TwoBridgePathableOnlyEnv(TwoBridgeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, include_player_relative=False, **kwargs)

