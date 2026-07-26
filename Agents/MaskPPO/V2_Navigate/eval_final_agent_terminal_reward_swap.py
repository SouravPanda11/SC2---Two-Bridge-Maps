from pathlib import Path
import multiprocessing as mp
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.MaskPPO._eval_terminal_reward_experiments import main_for_script
from Agents.terminal_reward_eval_common import EXPERIMENT_REWARD_SWAP


if __name__ == "__main__":
    mp.freeze_support()
    main_for_script(
        __file__,
        experiment=EXPERIMENT_REWARD_SWAP,
        final_only=True,
    )
