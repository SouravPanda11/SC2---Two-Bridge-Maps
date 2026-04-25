from __future__ import annotations

from typing import Sequence

import numpy as np

import Environments.MultiAgent._qmix_maskppo_base as qmix_base_module
from Environments.MultiAgent._qmix_maskppo_base import (
    SCR_RES,
    TwoBridgeMapConfig,
    TwoBridgeQMixMaskPPOEnvBase,
)


DEFAULT_MINIMAP_CROP = (28, 60, 0, 32)  # y0, y1, x0, x1
_REGISTERED_MAP_CLASSES = {}


def _register_two_bridge_map_once(config: TwoBridgeMapConfig) -> None:
    map_cls = _REGISTERED_MAP_CLASSES.get(config.registry_name)
    if map_cls is None:
        map_cls = type(
            config.registry_name,
            (qmix_base_module.lib.Map,),
            {
                "name": config.registry_name,
                "directory": config.directory,
                "filename": config.filename,
                "players": config.players,
            },
        )
        _REGISTERED_MAP_CLASSES[config.registry_name] = map_cls

    registered_maps = qmix_base_module.lib.get_maps()
    registered_maps.pop(config.registry_name, None)
    registered_maps[config.registry_name] = map_cls()


class TwoBridgeQMixReducedEnvBase(TwoBridgeQMixMaskPPOEnvBase):
    """
    QMIX environment with the same dynamics as MultiAgent QMIX, but with a
    smaller visual minimap input.

    The full PySC2 minimap is still requested at 64x64 so raw-unit action
    coordinates and existing environment logic stay unchanged. Only the
    exposed minimap tensor returned by get_minimap() is changed.
    """

    def __init__(
        self,
        *,
        map_config: TwoBridgeMapConfig,
        map_name: str,
        minimap_crop: Sequence[int] = DEFAULT_MINIMAP_CROP,
        include_player_relative: bool = True,
        **kwargs,
    ):
        self.minimap_crop = self._validate_crop(minimap_crop)
        self.include_player_relative = bool(include_player_relative)
        qmix_base_module.register_two_bridge_map = _register_two_bridge_map_once
        super().__init__(map_config=map_config, map_name=map_name, **kwargs)
        self._minimap = np.zeros(self.reduced_minimap_shape, dtype=np.uint8)

    @property
    def reduced_minimap_shape(self) -> tuple[int, int, int]:
        y0, y1, x0, x1 = self.minimap_crop
        channels = 2 if self.include_player_relative else 1
        return channels, y1 - y0, x1 - x0

    @staticmethod
    def _validate_crop(crop: Sequence[int]) -> tuple[int, int, int, int]:
        if len(crop) != 4:
            raise ValueError(
                "minimap_crop must be (y0, y1, x0, x1), for example "
                f"{DEFAULT_MINIMAP_CROP}."
            )

        y0, y1, x0, x1 = (int(value) for value in crop)
        if not (0 <= y0 < y1 <= SCR_RES and 0 <= x0 < x1 <= SCR_RES):
            raise ValueError(
                "Invalid minimap_crop. Expected 0 <= y0 < y1 <= 64 and "
                f"0 <= x0 < x1 <= 64, received {(y0, y1, x0, x1)}."
            )
        return y0, y1, x0, x1

    def _update_observations(self, ts):
        super()._update_observations(ts)
        self._minimap = self._reduce_minimap(self._minimap)

    def _reduce_minimap(self, full_minimap: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.minimap_crop
        reduced = np.asarray(full_minimap, dtype=np.uint8)[:, y0:y1, x0:x1]
        if not self.include_player_relative:
            reduced = reduced[:1]
        return np.ascontiguousarray(reduced)

    def get_env_info(self):
        info = super().get_env_info()
        info.update(
            {
                "minimap_shape": self.reduced_minimap_shape,
                "minimap_crop": tuple(int(value) for value in self.minimap_crop),
                "include_player_relative": bool(self.include_player_relative),
            }
        )
        return info
