# train_full.py ----------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import CombinedExtractor

from Environments.TB_env_AS20 import TwoBridgeEnv

# ---- config ------------------------------------------------------------
TOTAL_STEPS   = 1_000_000
SAVE_EVERY    = 250_000
RUN_NAME      = "SB_PPO_SF_AS20"
SAVE_DIR      = f"./Agents/saved_models/{RUN_NAME}"
TB_LOGDIR     = f"./tb_logs/{RUN_NAME}"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---- environment -------------------------------------------------------
env = TwoBridgeEnv(obs_type="full", visualize=False)

# ---- policy kwargs (CNN-head + MLP) ------------------------------------
policy_kwargs = dict(
    features_extractor_class = CombinedExtractor,
    features_extractor_kwargs= dict(cnn_output_dim=128),
)

# ---- model -------------------------------------------------------------
model = PPO(
    policy          = "MultiInputPolicy",
    env             = env,
    policy_kwargs   = policy_kwargs,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    verbose         = 1,
    tensorboard_log = TB_LOGDIR,
)

# ---- checkpoints -------------------------------------------------------
ckpt = CheckpointCallback(
    save_freq   = SAVE_EVERY,
    save_path   = SAVE_DIR,
    name_prefix = "ppo_full",
    save_replay_buffer = False,
    save_vecnormalize  = False,
)

# ---- train -------------------------------------------------------------
model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt)
model.save(f"{SAVE_DIR}/ppo_full_final")
env.close()
