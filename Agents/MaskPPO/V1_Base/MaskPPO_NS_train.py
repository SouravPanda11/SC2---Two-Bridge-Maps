import os
import random
import sys

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# Environment imports
from Environments.NS_AM_RM_mean.V1_Base_NS import TwoBridgeEnv, N_FRIEND, N_ENEMY


NUM_SEEDS = 3
TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL = 100_000

AGENT_NAME = "MaskPPO_NS_AM_RM_mean"
MAP_NAME = "V1_Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) ->
    MultiDiscrete([3, 2xN_FRIEND, 9, N_ENEMY+1])
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete([3] + [2] * N_FRIEND + [9] + [N_ENEMY + 1])

        # Bits beyond the verb-level mask that are always legal.
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        # Advertise flattened mask to SB3.
        flat_len = 3 + len(self._mask_template)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb": int(a_vec[0]),
            "who": np.asarray(a_vec[1 : 1 + N_FRIEND], np.int8),
            "direction": int(a_vec[1 + N_FRIEND]),
            "enemy_idx": int(a_vec[-1]),
        }

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        flat_mask = np.concatenate([obs["action_mask"], self._mask_template]).astype(np.int8)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask


class TBRewardLogger(BaseCallback):
    """
    Logs env-provided reward components under 'rew/*' in TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

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
        return True


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_random_seeds(num_seeds):
    rng = random.SystemRandom()
    seeds = []
    seen = set()

    while len(seeds) < num_seeds:
        seed = rng.randrange(0, 2**31 - 1)
        if seed in seen:
            continue
        seeds.append(seed)
        seen.add(seed)

    return tuple(seeds)


def mask_fn(env):
    return env.action_masks()


def validate_obs_contract(base_env, flat_env, obs):
    expected_keys = {"minimap", "vector", "action_mask"}
    if set(obs.keys()) != expected_keys:
        raise RuntimeError(f"Unexpected observation keys: {sorted(obs.keys())}")

    expected_minimap_shape = base_env.observation_space["minimap"].shape
    expected_vector_shape = base_env.observation_space["vector"].shape
    expected_mask_shape = flat_env.observation_space["action_mask"].shape

    if obs["minimap"].shape != expected_minimap_shape:
        raise RuntimeError(
            f"Unexpected minimap shape: {obs['minimap'].shape} != {expected_minimap_shape}"
        )
    if obs["vector"].shape != expected_vector_shape:
        raise RuntimeError(
            f"Unexpected vector shape: {obs['vector'].shape} != {expected_vector_shape}"
        )
    if obs["action_mask"].shape != expected_mask_shape:
        raise RuntimeError(
            f"Unexpected action_mask shape: {obs['action_mask'].shape} != {expected_mask_shape}"
        )


def make_env(seed):
    base_env = TwoBridgeEnv(visualize=False, realtime=False)
    flat_env = FlattenActionWrapper(base_env)
    env = ActionMasker(flat_env, mask_fn)

    obs, _ = env.reset(seed=seed)
    validate_obs_contract(base_env, flat_env, obs)

    print(
        "Obs contract OK | "
        f"seed={seed} | "
        f"keys={sorted(obs.keys())} | "
        f"minimap={obs['minimap'].shape} | "
        f"vector={obs['vector'].shape} | "
        f"action_mask={obs['action_mask'].shape}"
    )
    return env


def format_step_label(total_steps):
    if total_steps % 1000 == 0:
        return f"{total_steps // 1000}K"
    return str(total_steps)


def get_seed_output_dirs(seed):
    save_dir = os.path.join(
        project_root,
        "Agents",
        "MaskPPO",
        MAP_NAME,
        "saved_models",
        AGENT_NAME,
        f"seed_{seed}",
    )
    tb_log_dir = os.path.join(
        project_root,
        "tb_logs",
        "MaskPPO",
        MAP_NAME,
        AGENT_NAME,
        f"seed_{seed}",
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    return save_dir, tb_log_dir


def train_for_seed(seed):
    set_global_seeds(seed)
    save_dir, tb_log_dir = get_seed_output_dirs(seed)

    print(f"Starting training | device={DEVICE} | seed={seed}")
    print(f"Checkpoint dir: {save_dir}")
    print(f"TensorBoard dir: {tb_log_dir}")

    env = make_env(seed)

    try:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            device=DEVICE,
            verbose=1,
            tensorboard_log=tb_log_dir,
            seed=seed,
        )

        tb_callback = TBRewardLogger()
        timesteps_done = 0

        while timesteps_done < TOTAL_TIMESTEPS:
            step_chunk = min(SAVE_INTERVAL, TOTAL_TIMESTEPS - timesteps_done)
            model.learn(
                total_timesteps=step_chunk,
                reset_num_timesteps=False,
                callback=tb_callback,
                progress_bar=True,
                tb_log_name="train",
            )

            timesteps_done += step_chunk
            checkpoint_name = f"{AGENT_NAME}_{format_step_label(timesteps_done)}"
            model.save(os.path.join(save_dir, checkpoint_name))
            print(f"Saved checkpoint | seed={seed} | steps={timesteps_done}")

        model.save(os.path.join(save_dir, f"{AGENT_NAME}_final"))
        print(f"Finished training | seed={seed} | total_steps={TOTAL_TIMESTEPS}")
    finally:
        env.close()


def main():
    seeds = generate_random_seeds(NUM_SEEDS)
    print(
        f"Using device: {DEVICE} | "
        f"seeds={seeds} | "
        f"total_timesteps={TOTAL_TIMESTEPS} | "
        f"save_interval={SAVE_INTERVAL}"
    )

    for seed in seeds:
        train_for_seed(seed)


if __name__ == "__main__":
    main()
