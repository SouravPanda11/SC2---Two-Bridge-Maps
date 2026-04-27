from pathlib import Path
import multiprocessing as mp
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.MAPPO_reduced._train_mappo_reduced import train_with_settings


MAP_NAME = "V2_Combat"
RUN_MODE = "fresh_start"
# RUN_MODE = "load_last_checkpoint"

FRESH_START_SEED = None
FRESH_START_SEED_VALUES: tuple[int, ...] = ()

TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL = 50_000
NUM_SEEDS = 3
NUM_ENVS = 3

INCLUDE_PLAYER_RELATIVE = True
APPEND_MINIMAP_TO_STATE = True
APPEND_MINIMAP_TO_OBS = False
USE_TENSORBOARD = True
SMOKE_TEST = False


def main():
    train_with_settings(
        map_name=MAP_NAME,
        run_mode=RUN_MODE,
        seed=FRESH_START_SEED,
        seed_values=FRESH_START_SEED_VALUES,
        total_timesteps=TOTAL_TIMESTEPS,
        save_interval=SAVE_INTERVAL,
        num_seeds=NUM_SEEDS,
        num_envs=NUM_ENVS,
        include_player_relative=INCLUDE_PLAYER_RELATIVE,
        append_minimap_to_state=APPEND_MINIMAP_TO_STATE,
        append_minimap_to_obs=APPEND_MINIMAP_TO_OBS,
        use_tensorboard=USE_TENSORBOARD,
        smoke_test=SMOKE_TEST,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
