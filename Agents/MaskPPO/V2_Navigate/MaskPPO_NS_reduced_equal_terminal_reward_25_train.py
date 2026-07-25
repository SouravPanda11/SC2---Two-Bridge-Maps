from pathlib import Path
import multiprocessing as mp
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.MaskPPO._train_maskppo_reduced_equal_terminal_reward import (
    train_with_settings,
)


MAP_NAME = "V2_Navigate"

# Run mode: use fresh_start once, then switch to load_last_checkpoint after
# an interrupted run.
RUN_MODE = "fresh_start"
# RUN_MODE = "load_last_checkpoint"

# Matched to two seeds from the established V2 Navigate MaskPPO run.
FRESH_START_SEED_VALUES = (823800835, 788967690)

TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL = 50_000
NUM_SEEDS = 2
NUM_ENVS = 3


def main():
    train_with_settings(
        map_name=MAP_NAME,
        run_mode=RUN_MODE,
        seed_values=FRESH_START_SEED_VALUES,
        total_timesteps=TOTAL_TIMESTEPS,
        save_interval=SAVE_INTERVAL,
        num_seeds=NUM_SEEDS,
        num_envs=NUM_ENVS,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
