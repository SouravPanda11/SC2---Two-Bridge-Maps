from __future__ import annotations


NAV_WIN_BONUS = 25.0
COMBAT_WIN_BONUS = 25.0


class EqualTerminalSuccessRewardMixin:
    """
    Set both successful terminal rewards to 25 in an existing Two Bridge env.

    Dense navigation/combat shaping, observations, dynamics, loss penalties,
    and tie rewards continue to come from the original environment.
    """

    def _shape_reward(self, vec, done, res):
        had_reward_history = not (
            self._prev_enemy_hp.sum() == 0
            and self._prev_friend_hp.sum() == 0
        )
        reward = super()._shape_reward(vec, done, res)

        if not done or not had_reward_history:
            return reward

        if res == "nav_win":
            equal_term_r = NAV_WIN_BONUS
        elif res in {"combat_win", "victory"}:
            equal_term_r = COMBAT_WIN_BONUS
        else:
            return reward

        original_term_r = float(self._last_reward_components["term_r"])
        self._last_reward_components["term_r"] = float(equal_term_r)
        return float(reward - original_term_r + equal_term_r)
