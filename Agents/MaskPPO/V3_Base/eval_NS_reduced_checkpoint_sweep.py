from pathlib import Path
import multiprocessing as mp
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.MaskPPO._eval_ns_reduced_checkpoint_sweep import main_for_script


AGENT_NAME = "MaskPPO_NS_reduced"
# AGENT_NAME = "MaskPPO_NS_reduced_pathable_only"


if __name__ == "__main__":
    mp.freeze_support()
    main_for_script(__file__, default_agent_name=AGENT_NAME)
