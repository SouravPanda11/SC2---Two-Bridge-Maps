# train_vec.py -----------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from Environments.TB_env_AS20 import TwoBridgeEnv

# ---- config ------------------------------------------------------------
TOTAL_STEPS   = 10      # 1 M
SAVE_EVERY    = 3        # save 4× during training
RUN_NAME      = "SB_PPO_NSF_AS20"      # folder + tensorboard tag
SAVE_DIR      = f"./Agents/saved_models/{RUN_NAME}"
TB_LOGDIR     = f"./tb_logs/{RUN_NAME}"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---- environment -------------------------------------------------------
env = TwoBridgeEnv(obs_type="vector", visualize=False)

# ---- model -------------------------------------------------------------
model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    device          = "cpu", #since using MLP policy
    verbose         = 1,
    tensorboard_log = TB_LOGDIR,
)

# ---- checkpoints -------------------------------------------------------
ckpt = CheckpointCallback(
    save_freq   = SAVE_EVERY,
    save_path   = SAVE_DIR,
    name_prefix = "ppo_vec",
    save_replay_buffer = False,
    save_vecnormalize  = False,
)

# ---- train -------------------------------------------------------------
model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt)
model.save(f"{SAVE_DIR}/ppo_vec_final")
env.close()
