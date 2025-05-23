# Two-Bridge Benchmark
This repository accompanies the paper(will be added after review cycle) by providing the **reinforcement-learning (RL) training code** used to generate the empirical results reported in our study. While the benchmark itself is fully self-contained within the StarCraft II map triggers, the scripts here wrap those custom maps with PySC2-based Gym environments and train agents with Stable-Baselines 3.

Consequently, these files are **not required for running or evaluating the maps**; they serve purely to document and reproduce the RL experiments showcased in the paper.

## Repository Overview

```
SC2-Two-Bridge-Maps/
├── Agents/                         # Training / evaluation entry-points
│   ├── *train.py                   # SB3 scripts (A2C, PPO, Maskable-PPO …)
│   ├── Agent Performance Charts/   # PNG win-rate curves
│   └── saved_models/               # model checkpoints & “final.zip”
│
├── Environments/                   # Gymnasium + PySC2 envs per map variant
│   └── TB_env<variant>.py
│
├── Maps/                           # *.SC2Map files (drop into SC2/Maps)
│   └── TwoBridgeMap<Vi>_<mode>.SC2Map
│
├── tb_logs/                        # TensorBoard event files
├── register_bridge_map.py          # Utility: adds map to PySC2 registry
├── requirements.txt                # Python dependencies 
└── README.md
```

### Naming convention

| Token | Meaning |
|-------|---------|
| **SB**        | _Stable-Baselines3_ implementation |
| **A2C / PPO / MaskPPO** | RL algorithm |
| **NSF**       | _No Spatial Features_: vector-only observation |
| **SF**        | _Spatial Features_: adds 64×64 screen + minimap |
| **AS**        | _Action space_ |
| **AM**        | _Action Masking_ enabled |
| **V1 / V2 / V3** | Unit based map variants |
| **Base / navigate / combat** | Objective placement based map variants |

## Quick-Start Guide

1. Download and install StarCraft II from Battlenet.
2. Create Python env & install dependencies.
3. Copy the maps into SC2/Maps folder.
4. Train an agent.
5. Watch progress in TensorBoard:
6. Evaluate
--- 

### Repository contents at a glance

- **Environments/** – Each `TB_env<variant>.py` wraps a custom map as a Gymnasium env (hybrid multi-discrete actions, optional action-mask).
- **Agents/** – `*train.py` scripts that launch Stable-Baselines 3 runs and log to TensorBoard.
- **Pre-trained policies** – All checkpoints (`final.zip` plus intermediates) live in `Agents/saved_models/<run-id>/`; load them with the supplied `eval_*` scripts to reproduce the paper’s results or watch qualitative behaviour.
- **Performance curves** – Win-rate PNGs are in `Agents/Agent Performance Charts/`.

**Note:**  
All scripts and environments have been tested on Windows only. Linux support is not guaranteed and has not been tested.

## Pre-Trained Agents

| Algorithm | Obs. Space | Action Space | Map Variant | Checkpoints |
|-----------|------------|-------------|-------------|-------------|
| **A2C**   | SF / NSF   | 14       | V2-Base | 400 K → 2 M |
| **PPO**   | SF / NSF   | 14       | V2-Base | 100 K → 1 M(NSF) / 400 k → 2M(SF)|
| **Mask-PPO** | SF      | AM       | V2-Base, V3-navigate, V3-combat | 400 K → 2 M |

## Acknowledgements

PySC2 by DeepMind  
Stable-Baselines3 contributors  
Blizzard Entertainment (for SC2)  
StarCraft community for inspiration

This is an open-source educational project. Not affiliated with Blizzard.