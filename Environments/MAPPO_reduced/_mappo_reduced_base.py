from __future__ import annotations

from Environments.QMIX_reduced._qmix_reduced_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixReducedEnvBase,
)


class TwoBridgeMAPPOReducedEnvBase(TwoBridgeQMixReducedEnvBase):
    """
    MAPPO-facing reduced Two Bridge environment.

    The environment dynamics, action availability, reward shaping, and reduced
    minimap contract intentionally match the QMIX reduced pipeline. The MAPPO
    EPyMARL wrapper decides how to fold the reduced minimap into EPyMARL's
    flat observation/state tensors.
    """

    pass
