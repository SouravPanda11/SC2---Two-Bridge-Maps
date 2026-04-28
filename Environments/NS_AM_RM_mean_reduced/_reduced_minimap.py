from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_MINIMAP_CROP = (28, 60, 0, 32)  # y0, y1, x0, x1
REDUCED_MINIMAP_SHAPE = (2, 32, 32)
PATHABLE_ONLY_MINIMAP_SHAPE = (1, 32, 32)


def validate_minimap_crop(crop: Sequence[int]) -> tuple[int, int, int, int]:
    if len(crop) != 4:
        raise ValueError("minimap_crop must be (y0, y1, x0, x1).")

    y0, y1, x0, x1 = (int(value) for value in crop)
    if not (0 <= y0 < y1 <= 64 and 0 <= x0 < x1 <= 64):
        raise ValueError(
            "Invalid minimap_crop. Expected 0 <= y0 < y1 <= 64 and "
            f"0 <= x0 < x1 <= 64, received {(y0, y1, x0, x1)}."
        )
    if (y1 - y0, x1 - x0) != (32, 32):
        raise ValueError(
            "MaskPPO reduced env expects a 32x32 crop, received "
            f"{(y1 - y0, x1 - x0)}."
        )
    return y0, y1, x0, x1


def reduced_minimap_shape(include_player_relative: bool = True) -> tuple[int, int, int]:
    return REDUCED_MINIMAP_SHAPE if include_player_relative else PATHABLE_ONLY_MINIMAP_SHAPE


def reduce_minimap(
    minimap: np.ndarray,
    crop: Sequence[int] = DEFAULT_MINIMAP_CROP,
    include_player_relative: bool = True,
) -> np.ndarray:
    y0, y1, x0, x1 = validate_minimap_crop(crop)
    reduced = np.asarray(minimap, dtype=np.uint8)[:, y0:y1, x0:x1]
    if not include_player_relative:
        reduced = reduced[:1]
    return np.ascontiguousarray(reduced)
