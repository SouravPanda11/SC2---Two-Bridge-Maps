import os
import sys
import numpy as np
import torch as th

# ───────────────── project root ─────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ───────────────── imports ──────────────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Env import (BEACON-ONLY VERSION) – same as your train script
from Environments.full_action_mask.TB_env_FAM_V2_Base_Beacon_Fixed import (
    TwoBridgeEnv, N_FRIEND
)

# ───────────────── FlattenActionWrapper (no enemy) ─────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction) →
    MultiDiscrete([2, 2×N_FRIEND, 9])

    Flat action layout:
      [ verb(2),
        who_0(2), who_1(2), ..., who_{N_FRIEND-1}(2),
        direction(9) ]
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

        verb_mask      = np.asarray(am["verb"], dtype=np.int8)      # (2,)
        who_bits       = np.asarray(am["who"], dtype=np.int8)       # (N_FRIEND,)
        direction_mask = np.asarray(am["direction"], np.int8)       # (9,)

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


def mask_fn(env):
    return env.action_masks()


def count_params(module: th.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ──────────── build env exactly as in train script ────────────
    base_env = TwoBridgeEnv(visualize=False)
    flat_env = FlattenActionWrapper(base_env)
    env      = ActionMasker(flat_env, mask_fn)

    # ──────────── build model (no training) ────────────
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=1,
    )

    print("\n=== FULL POLICY ARCHITECTURE ===")
    print(model.policy)

    print("\n=== FEATURES EXTRACTOR (CombinedExtractor) ===")
    fe = model.policy.features_extractor
    print(fe)

    # If it's CombinedExtractor, it has per-key extractors
    if hasattr(fe, "extractors"):
        print("\nPer-input extractors:")
        for key, sub in fe.extractors.items():
            print(f"  [{key}] →")
            print(f"    {sub}")

    # ──────────── parameter counts ────────────
    total_params = count_params(model.policy)
    fe_params    = count_params(model.policy.features_extractor)
    pi_head      = model.policy.action_net
    vf_head      = model.policy.value_net
    pi_params    = count_params(pi_head)
    vf_params    = count_params(vf_head)

    print("\n=== PARAMETER COUNTS ===")
    print(f"Total policy params:        {total_params}")
    print(f"Feature extractor params:   {fe_params}")
    print(f"Policy head (action_net):   {pi_params}")
    print(f"Value head  (value_net):    {vf_params}")

    # Optional: show observation space and action space for sanity
    print("\nObs space:", env.observation_space)
    print("Action space:", env.action_space)

    env.close()
