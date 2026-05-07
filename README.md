# Two-Bridge Benchmark (StarCraft II)

Two-Bridge is a StarCraft II reinforcement-learning benchmark for studying
navigation and micro-combat without the compute cost of full-game SC2. The map
logic lives in the `.SC2Map` trigger files; this repository provides the Python
environment wrappers, training scripts, evaluation scripts, checkpoints,
analysis outputs, and the static project page used for the benchmark.

These scripts are not required to play the maps in StarCraft II. They are for
reproducing and extending the RL experiments.

## Repository layout

```text
.
|-- Agents/
|   |-- A2C/                 # Original Stable-Baselines3 A2C scripts
|   |-- PPO/                 # Original Stable-Baselines3 PPO scripts
|   |-- MaskPPO/             # Maskable PPO variants and checkpoint sweeps
|   |-- MAPPO_reduced/       # Custom reduced-observation MAPPO pipeline
|   |-- Qmix_reduced/        # Custom reduced-observation QMIX pipeline
|   |-- checkpoint_sweep_eval_common.py
|   |-- run_A2C_PPO_evals.py
|-- Environments/
|   |-- Pilot/               # Original V2 Base A2C/PPO envs
|   |-- AM_RM_mean/          # Camera-free MaskPPO envs with spatial features
|   |-- FAM_CAM/             # Camera-lock MaskPPO envs and mutant envs
|   |-- NS_AM_RM_mean/       # No-screen per-unit MaskPPO envs
|   |-- NS_AM_RM_mean_reduced/
|   |-- MAPPO_reduced/
|   |-- QMIX_reduced/
|   |-- Utilities/           # Map-bound and minimap-crop inspection tools
|-- Maps/
|   |-- Camera Free/
|   |-- Camera Lock/
|-- Agent Performance Charts/
|-- docs/                    # Static GitHub Pages project/paper page
|-- tb_logs/                 # TensorBoard outputs
|-- register_bridge_map.py
|-- requirements.txt
|-- README.md
```

## Prerequisites

- Windows. Several environment files contain hardcoded Windows StarCraft II map
  paths.
- StarCraft II installed from Battle.net and launched at least once.
- Python with the dependencies in `requirements.txt`.
- Enough disk space for the included checkpoints, plots, TensorBoard logs, and
  generated replay outputs.

## Setup

### 1. Install Python dependencies

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you use a different virtual environment, run all commands below from the
repository root so imports such as `Agents.*` and `Environments.*` resolve.

### 2. Install the SC2 maps

Copy the camera-free maps:

```text
Maps/Camera Free/*.SC2Map
```

to:

```text
C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free
```

Copy the camera-lock maps:

```text
Maps/Camera Lock/*.SC2Map
Maps/Camera Lock/Mutants/*.SC2Map
```

to:

```text
C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Lock
C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Lock/Mutants
```

If your SC2 installation uses another location, update the `directory` field in
the relevant environment module under `Environments/`.

### 3. Optional map registration helper

Most workflows register their map class when the environment module is imported.
`register_bridge_map.py` is a small helper for the original V2 Base map:

```powershell
python register_bridge_map.py
```

## Map variants

| Variant | Enemies | Notes |
|---|---:|---|
| `V1_Base`, `V1_Combat`, `V1_Navigate` | 3 | Smallest enemy count |
| `V2_Base`, `V2_Combat`, `V2_Navigate` | 5 | Medium enemy count |
| `V3_Base`, `V3_Combat`, `V3_Navigate` | 8 | Largest enemy count |

All variants use five friendly marines. Episodes can end as `nav_win`,
`combat_win`, `combat_loss`, `timeout_loss`, or `tie`, with some legacy scripts
also collapsing PySC2 `victory` / `defeat` labels into combat outcomes.

## Training

### Original A2C/PPO baselines

```powershell
python Agents\A2C\SB_A2C_NSF_AS14_train.py
python Agents\A2C\SB_A2C_SF_AS14_train.py
python Agents\PPO\SB_PPO_NSF_AS14_train.py
python Agents\PPO\SB_PPO_SF_AS14_train.py
```

These use the original `Pilot/` V2 Base environments. Final checkpoints are
saved under `Agents/A2C/saved_models/` and `Agents/PPO/saved_models/`.

### Original MaskPPO variants

Each map variant has its own training script under `Agents/MaskPPO/<variant>/`:

```powershell
python Agents\MaskPPO\V2_Base\SB_MPPO_SF_AM_RM_mean_train.py
python Agents\MaskPPO\V2_Base\SB_MPPO_FAM_CAM_train.py
```

The `AM_RM_mean` scripts use camera-free maps. The `FAM_CAM` scripts use
camera-lock maps. These legacy MaskPPO runs typically train for 5,000,000
timesteps with 500,000-step checkpoints.

### Reduced MaskPPO

Reduced MaskPPO uses per-unit action masking and a smaller minimap input. Each
map variant has `MaskPPO_NS_parallel_train.py` and
`MaskPPO_NS_reduced_parallel_train.py` scripts:

```powershell
python Agents\MaskPPO\V1_Base\MaskPPO_NS_reduced_parallel_train.py
```

Defaults are generally three seeds, three parallel SC2 environments,
2,000,000 timesteps, and 50,000-step checkpoint targets. The reduced minimap
crop is configured in `Environments/NS_AM_RM_mean_reduced/_reduced_minimap.py`
and currently uses a 32x32 crop from the 64x64 PySC2 minimap.

### MAPPO reduced

MAPPO reduced scripts live under `Agents/MAPPO_reduced/<variant>/`:

```powershell
python Agents\MAPPO_reduced\V2_Base\train_mappo.py
```

The shared implementation is `Agents/MAPPO_reduced/_train_mappo_reduced.py`.
Scripts expose constants near the top for `RUN_MODE`, seeds, number of envs,
total timesteps, minimap usage, and whether player-relative minimap channels are
included.

### QMIX reduced

QMIX reduced scripts live under `Agents/Qmix_reduced/<variant>/`:

```powershell
python Agents\Qmix_reduced\V2_Base\train_qmix.py
```

QMIX checkpoints are PyTorch `.pt` files. By default these scripts train pathable
only variants (`QMIX_reduced_pathable_only`) unless `INCLUDE_PLAYER_RELATIVE` is
changed in the script.

## Evaluation

### Original A2C/PPO

```powershell
python Agents\PPO\eval_PPO_SF_AS14_agent.py --episodes 100
python Agents\run_A2C_PPO_evals.py
```

### Original MaskPPO

```powershell
python Agents\MaskPPO\V2_Base\eval_AM_RM_mean_agent.py --episodes 100
python Agents\MaskPPO\V2_Base\eval_FAM_CAM_agent.py --episodes 100
python Agents\MaskPPO\run_AM_RM_mean_evals.py
python Agents\MaskPPO\run_FAM_CAM_evals.py
```

These scripts write plots and JSON/TXT summaries under
`Agent Performance Charts/` and may write `.SC2Replay` files under `Replays/`.

### Reduced checkpoint sweeps

Reduced MaskPPO, MAPPO, and QMIX use checkpoint-sweep evaluators that read the
latest seed manifest, evaluate each step checkpoint, cache CSV rows, and produce
win-rate plots.

```powershell
python Agents\MaskPPO\V2_Base\eval_NS_reduced_checkpoint_sweep.py --episodes 32 --num-eval-envs 16
python Agents\MAPPO_reduced\V2_Base\eval_checkpoint_sweep.py --episodes 32 --num-eval-envs 16
python Agents\Qmix_reduced\V2_Base\eval_checkpoint_sweep.py --episodes 32 --num-eval-envs 16
```

Useful options:

```text
--map-name <variant>        # Required only when running shared evaluator modules directly
--agent-name <name>         # Evaluate pathable-only or player-relative checkpoint folders
--device auto|cpu|cuda
--overwrite                 # Ignore cached CSV rows
--stochastic                # Use stochastic policy evaluation where supported
```

QMIX also supports `--epsilon` during evaluation.

## Outputs

- Checkpoints:
  - `Agents/A2C/saved_models/`
  - `Agents/PPO/saved_models/`
  - `Agents/MaskPPO/<variant>/saved_models/`
  - `Agents/MAPPO_reduced/<variant>/saved_models/`
  - `Agents/Qmix_reduced/<variant>/saved_models/`
- TensorBoard logs:
  - `tb_logs/`
- Evaluation plots, CSVs, and metadata:
  - `Agent Performance Charts/`
- Optional SC2 replay files:
  - `Replays/`

Run TensorBoard with:

```powershell
tensorboard --logdir tb_logs
```

## Analysis and plotting

Aggregate plotting scripts live in `Agent Performance Charts/`:

```powershell
python "Agent Performance Charts\PPO vs A2C overall plots.py"
python "Agent Performance Charts\MPPO_AM_RM_mean overall plots.py"
python "Agent Performance Charts\MPPO_FAM_CAM overall plots.py"
python "Agent Performance Charts\reduced_agents_multiplot.py"
python "Agent Performance Charts\reduced_agents_terminal_outcomes.py"
python "Agent Performance Charts\reduced_agents_mean_stacked_win_conditions.py"
```

Environment inspection utilities live in `Environments/Utilities/`:

```powershell
python Environments\Utilities\inspect_playable_area.py
python Environments\Utilities\plot_all_map_bounds_comparison.py
python Environments\Utilities\plot_smac_map_bounds.py --include both
python Environments\Utilities\find_v2_base_minimap_crop_bounds.py
```

Some utility scripts assume local ignored copies of SMAC/SMACv2 or a local
working virtual environment.

## Naming guide

| Token | Meaning |
|---|---|
| `SB` | Stable-Baselines3 script |
| `A2C`, `PPO`, `MaskPPO` | Single-policy baseline algorithms |
| `MAPPO_reduced` | Custom multi-agent PPO implementation with reduced minimap |
| `QMIX_reduced` | Custom QMIX implementation with reduced minimap |
| `NSF` | No spatial features |
| `SF` | Spatial features |
| `NS` | No screen observation; compact vector/minimap setup |
| `AM` | Action masking |
| `FAM_CAM` | Full action masking on camera-lock maps |
| `AM_RM_mean` | Masked-action, reward-model mean environment family |
| `pathable_only` | Reduced minimap with only the pathable channel |
| `V1`, `V2`, `V3` | Increasing enemy-count map versions |
| `Base`, `Combat`, `Navigate` | Objective/layout variants |

## Notes and limitations

- The codebase is Windows-oriented and many scripts hardcode
  `C:/Program Files (x86)/StarCraft II/...`.
- Parallel SC2 training/evaluation starts several game clients. Reduce
  `NUM_ENVS` or `--num-eval-envs` if startup is unstable or memory is tight.
- Some older environment files contain display/comment encoding artifacts, but
  the executable constants and logic are still plain Python.
- The repository includes generated artifacts and checkpoints. A fresh clone can
  be large.
- Generated `Replays/` outputs are not tracked by default.

## Acknowledgements

This project builds on PySC2, Stable-Baselines3, sb3-contrib, PyTorch, Gymnasium,
and StarCraft II. StarCraft II is owned by Blizzard Entertainment. This is an
open-source educational/research project and is not affiliated with Blizzard.
