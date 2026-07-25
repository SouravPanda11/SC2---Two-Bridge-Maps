from Environments.NS_AM_RM_mean_reduced.V2_Navigate_NS import *  # noqa: F401,F403
from Environments.NS_AM_RM_mean_reduced.V2_Navigate_NS import (
    TwoBridgeEnv as _OriginalReducedEnv,
)
from Environments.NS_AM_RM_mean_reduced._equal_terminal_reward import (
    EqualTerminalSuccessRewardMixin,
)


class TwoBridgeEnv(EqualTerminalSuccessRewardMixin, _OriginalReducedEnv):
    """V2 Navigate reduced MaskPPO environment with 25/25 win rewards."""

    pass


class TwoBridgePathableOnlyEnv(TwoBridgeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, include_player_relative=False, **kwargs)
