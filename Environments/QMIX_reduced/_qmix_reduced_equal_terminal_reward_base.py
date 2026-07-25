from __future__ import annotations

from Environments.QMIX_reduced._qmix_reduced_base import (
    TwoBridgeMapConfig,
    TwoBridgeQMixReducedEnvBase,
)


NAV_WIN_BONUS = 25.0
COMBAT_WIN_BONUS = 25.0


class TwoBridgeQMixEqualTerminalRewardReducedEnvBase(
    TwoBridgeQMixReducedEnvBase
):
    """
    Reduced environment used only by the equal-success-reward experiment.

    Dense navigation/combat shaping and all loss/tie terminal rewards are
    inherited unchanged. Only the two successful terminal outcomes are set
    to the same value: navigation receives 25 and combat receives 25.
    """

    def _shape_reward(self, vec, done, result):
        had_reward_history = not (
            self._prev_enemy_hp.sum() == 0
            and self._prev_friend_hp.sum() == 0
        )
        reward = super()._shape_reward(vec, done, result)

        if not done or not had_reward_history:
            return reward

        if result == "nav_win":
            equal_term_r = NAV_WIN_BONUS
        elif result in {"combat_win", "victory"}:
            equal_term_r = COMBAT_WIN_BONUS
        else:
            return reward

        original_term_r = float(self._last_reward_components["term_r"])
        self._last_reward_components["term_r"] = float(equal_term_r)
        return float(reward - original_term_r + equal_term_r)
