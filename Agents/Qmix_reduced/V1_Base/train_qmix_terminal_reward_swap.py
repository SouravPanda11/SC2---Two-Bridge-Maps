from pathlib import Path
import multiprocessing as mp
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.Qmix_reduced._train_qmix_reduced_reward_swap import (
    train_with_settings,
)


MAP_NAME = "V1_Base"

# Matched to two seeds from the established V1_Base QMIX experiment.
FRESH_START_SEED_VALUES = (297975710, 1076891414)

TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL = 50_000
NUM_SEEDS = 2
NUM_ENVS = 3


def main():
    train_with_settings(
        map_name=MAP_NAME,
        run_mode="fresh_start",
        seed_values=FRESH_START_SEED_VALUES,
        total_timesteps=TOTAL_TIMESTEPS,
        save_interval=SAVE_INTERVAL,
        num_seeds=NUM_SEEDS,
        num_envs=NUM_ENVS,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
