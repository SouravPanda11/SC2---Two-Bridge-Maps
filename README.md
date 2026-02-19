# Two-Bridge Benchmark (StarCraft II)

This repository contains the reinforcement learning (RL) code used to train and evaluate agents on the Two-Bridge StarCraft II benchmark maps.

The benchmark logic itself lives in the map triggers. The Python code here provides PySC2 + Gymnasium wrappers, Stable-Baselines3 training/evaluation scripts, and analysis tooling used for experiments.

Important: these scripts are not required to play the maps in StarCraft II. They are for reproducing and extending the RL experiments.

## What is in this repository

- Custom SC2 map files for camera-free and camera-lock variants (`Maps/`)
- Gymnasium environment wrappers over PySC2 (`Environments/`)
- Baseline training and evaluation scripts for A2C, PPO, and Maskable PPO (`Agents/`)
- Pretrained checkpoints (`Agents/**/saved_models/*_final.zip`)
- TensorBoard logs and evaluation artifacts (`tb_logs/`, `Agent Performance Charts/`, `Replays/`)
- Mutant-agent generation and analysis workflow (`Mutant Agents/`)

## Repository layout

```text
.
|-- Agents/
|   |-- A2C/
|   |-- PPO/
|   |-- MaskPPO/
|   |-- run_A2C_PPO_evals.py
|-- Environments/
|   |-- Pilot/            (PPO/A2C envs; V2_Base)
|   |-- AM_RM_mean/       (MaskPPO envs; camera-free maps)
|   |-- FAM_CAM/          (MaskPPO envs; camera-lock maps)
|-- Maps/
|   |-- Camera Free/
|   |-- Camera Lock/
|-- Mutant Agents/
|-- Agent Performance Charts/
|-- Replays/
|-- tb_logs/
|-- register_bridge_map.py
|-- requirements.txt
|-- README.md
```

## Setup

### 1. Install StarCraft II (Windows)

Install the free StarCraft II client from Battle.net and launch it once.

### 2. Copy map files

Copy the `.SC2Map` files from:

- `Maps/Camera Free/`
- `Maps/Camera Lock/`

into your StarCraft II maps directory. The environment files expect maps under:

- `C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free`
- `C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Lock`

If your SC2 install path differs, update the `directory` field in the corresponding environment map registration class.

### 3. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Optional map registration helper

```powershell
python register_bridge_map.py
```

Note: environment modules also register maps when imported.

## Quick start

### Reproduce baseline training

A2C (SF, AS14):

```powershell
python Agents\A2C\SB_A2C_SF_AS14_train.py
```

PPO (SF, AS14):

```powershell
python Agents\PPO\SB_PPO_SF_AS14_train.py
```

MaskPPO (example: V2 Base, AM_RM_mean):

```powershell
python Agents\MaskPPO\V2_Base\SB_MPPO_SF_AM_RM_mean_train.py
```

TensorBoard logs are written to `tb_logs/`.

### Evaluate trained or pretrained models

Single script example:

```powershell
python Agents\PPO\eval_PPO_SF_AS14_agent.py --episodes 100
```

Run all A2C/PPO eval scripts:

```powershell
python Agents\run_A2C_PPO_evals.py
```

Run MaskPPO eval sets:

```powershell
python Agents\MaskPPO\run_AM_RM_mean_evals.py
python Agents\MaskPPO\run_FAM_CAM_evals.py
```

Evaluation outputs are saved under `Agent Performance Charts/` and optional replays under `Replays/`.

## Pretrained agents

Pretrained final checkpoints are included in:

- `Agents/A2C/saved_models/`
- `Agents/PPO/saved_models/`
- `Agents/MaskPPO/*/saved_models/`

Typical training budgets used in scripts:

- A2C/PPO: 2,000,000 timesteps
- MaskPPO: 5,000,000 timesteps

## Naming convention

| Token | Meaning |
|---|---|
| `SB` | Stable-Baselines3 implementation |
| `A2C` / `PPO` / `MaskPPO` | RL algorithm |
| `NSF` | No spatial features (vector-only observation) |
| `SF` | Spatial features (screen + minimap + vector) |
| `AS` | Action space |
| `AM` | Action masking enabled |
| `V1` / `V2` / `V3` | Map difficulty/version |
| `Base` / `Navigate` / `Combat` | Objective variant |
| `FAM_CAM` | Full action masking with camera-lock maps |
| `AM_RM_mean` | Masked action setup with reward-model variant |

## Mutant-agent workflow

Generate mutants from a base MaskPPO policy:

```powershell
python "Mutant Agents\generate_mutant.py"
```

Batch-evaluate generated mutants:

```powershell
python "Mutant Agents\eval_mutants.py"
```

Specialized analyses for map-specific mutants:

```powershell
python "Mutant Agents\evalYellow.py"
python "Mutant Agents\evalPink.py"
```

Mutant outputs are written to `Mutant Agents/performance/` and `Mutant Agents/replays/`.

## Notes and limitations

- This codebase has been tested on Windows only.
- Some scripts include hardcoded Windows paths and map names.
- `register_bridge_map.py` currently registers one map helper class; most workflows rely on map registration inside environment modules.
- If maps are not found, first check map filenames and directory locations expected by the selected environment file.

## Acknowledgements

- PySC2 (DeepMind)
- Stable-Baselines3 and sb3-contrib contributors
- Blizzard Entertainment (StarCraft II)

Open-source educational project, not affiliated with Blizzard.
