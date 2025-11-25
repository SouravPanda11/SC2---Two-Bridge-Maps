import sys, os, torch, numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# ───────────────── Env import (BEACON-ONLY VERSION) ───────────────
from Environments.full_action_mask.TB_env_FAM_V2_Base_Beacon_Fixed import (
    TwoBridgeEnv, N_FRIEND
)
# NOTE: no N_ENEMY here – your beacon env doesn’t define it.

# ──────────────────── FLATTEN-ACTION WRAPPER (NO ENEMY) ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction) →
    MultiDiscrete([2, 2×N_FRIEND, 9])

    Flat action layout:
      [ verb(2),
        who_0(2), who_1(2), ..., who_{N_FRIEND-1}(2),
        direction(9) ]

    For each who_i, the two entries correspond to choices {0,1}.
    """

    def __init__(self, env):
        super().__init__(env)

        # MultiDiscrete layout: [verb] + [who bits] + [direction]
        self.action_space = spaces.MultiDiscrete(
            [2] + [2] * N_FRIEND + [9]
        )

        # Build the observation space: keep everything from env, but
        # advertise a *flat* action_mask whose length is sum(nvec)
        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

        self._last_mask = np.ones(flat_len, dtype=np.int8)

    @staticmethod
    def _flatten(a_dict):
        # Not actually used by MaskablePPO, but keeping for completeness
        return np.array([
            a_dict["verb"],
            *a_dict["who"],
            a_dict["direction"],
        ], dtype=np.int64)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb":      int(a_vec[0]),
            "who":       np.asarray(a_vec[1 : 1 + N_FRIEND], np.int8),
            "direction": int(a_vec[1 + N_FRIEND]),
        }

    def _convert_mask(self, obs):
        """
        Convert env's dict masks → flat vector compatible with ActionMasker.
        Expected obs["action_mask"] structure from env:
          {
            "verb":      (2,),
            "who":       (N_FRIEND,),
            "direction": (9,),
          }
        """
        am = obs["action_mask"]  # dict of np.int8 arrays

        verb_mask = np.asarray(am["verb"], dtype=np.int8)        # (2,)
        who_bits  = np.asarray(am["who"], dtype=np.int8)         # (N_FRIEND,)
        direction_mask = np.asarray(am["direction"], np.int8)    # (9,)

        # For each who_i (∈ {0,1}), make a 2-entry mask: [allow_0, allow_1]
        who_pairs = []
        for b in who_bits:
            # 0 (don't select) always valid; 1 valid iff unit alive (b==1)
            who_pairs.extend([1, int(b)])

        flat_mask = np.concatenate([
            verb_mask,                      # 2
            np.asarray(who_pairs, np.int8), # 2*N_FRIEND
            direction_mask,                 # 9
        ], dtype=np.int8)

        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    # ActionMasker will call this to fetch the current mask
    def action_masks(self):
        return self._last_mask


# ───────────────────── TB CALLBACK (reward components) ─────────────────────
class TBRewardLogger(BaseCallback):
    """
    Logs env-provided reward components under 'rew/*' in TensorBoard.
    Also logs simple summaries of per-unit metrics from get_unit_metrics().
    Requires:
      - info['rew'] = env.get_reward_components()
      - base env implements get_unit_metrics()
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _unwrap_base_env(self):
        """
        Get the underlying TwoBridgeEnv from SB3's VecEnv + our wrappers.
        """
        env = self.training_env
        # SB3: training_env is usually a VecEnv; grab first sub-env
        if hasattr(env, "envs"):
            env = env.envs[0]
        # unwrap through ActionMasker / FlattenActionWrapper, etc.
        while hasattr(env, "env"):
            env = env.env
        return env

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        # ----- log team-level reward components from info['rew'] -----
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and "rew" in info and self.logger is not None:
                    for k, v in info["rew"].items():
                        try:
                            self.logger.record(f"rew/{k}", float(v))
                        except Exception:
                            pass
        elif isinstance(infos, dict) and "rew" in infos and self.logger is not None:
            for k, v in infos["rew"].items():
                try:
                    self.logger.record(f"rew/{k}", float(v))
                except Exception:
                    pass

        # ----- log per-unit summaries from get_unit_metrics() --------
        try:
            base = self._unwrap_base_env()
            if hasattr(base, "get_unit_metrics"):
                um = base.get_unit_metrics()  # {'friend': {tag: {...}}}
                f_dict = um.get("friend", {})

                if f_dict and self.logger is not None:
                    nav_rs    = [m.get("nav_r", 0.0) for m in f_dict.values()]
                    nav_dists = [m.get("nav_dist", 0.0) for m in f_dict.values()]
                    hps       = [m.get("hp", 0.0) for m in f_dict.values()]

                    # simple aggregate stats; avoids spamming TB with per-tag series
                    self.logger.record("units/nav_r_mean",    float(np.mean(nav_rs)))
                    self.logger.record("units/nav_r_max",     float(np.max(nav_rs)))
                    self.logger.record("units/nav_dist_mean", float(np.mean(nav_dists)))
                    self.logger.record("units/hp_mean",       float(np.mean(hps)))
        except Exception:
            # never break training because of logging
            pass

        return True


# ───────────────────── hardware / logging ─────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

agent_name = "SB_MaskPPO_FAM_Beacon_Fixed"
map_name   = "V2_Base"
save_dir   = f"./Agents/MaskPPO/{map_name}/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

tb_log_dir = f"./tb_logs/MaskPPO/{map_name}/{agent_name}/"
os.makedirs(tb_log_dir, exist_ok=True)

# ───────────────── env + wrappers ─────────────────────────────
def mask_fn(env):  # env is the FlattenActionWrapper
    return env.action_masks()

base_env = TwoBridgeEnv(visualize=False)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

# ───────────────────── model (Maskable) ───────────────────────
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    device=device,
    verbose=1,
    tensorboard_log=tb_log_dir
)

# ───────────────────── training loop ──────────────────────────
total_timesteps = 1_500_000   # 1.5 M
save_interval   = 500_000     # every 500 k
# total_timesteps = 10
# save_interval   = 3

tb_callback = TBRewardLogger()

for i in range(0, total_timesteps, save_interval):
    model.learn(
        total_timesteps=save_interval,
        reset_num_timesteps=False,
        callback=tb_callback,
        progress_bar=True
    )
    model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")

model.save(f"{save_dir}{agent_name}_final")
env.close()
