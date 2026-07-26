from __future__ import annotations

import json
import zipfile
from pathlib import Path


EXPERIMENT_REWARD_SWAP = "reward_swap"
EXPERIMENT_EQUAL_25 = "equal_terminal_reward_25"
SUPPORTED_EXPERIMENTS = (
    EXPERIMENT_REWARD_SWAP,
    EXPERIMENT_EQUAL_25,
)


def validate_experiment(experiment: str) -> str:
    experiment = str(experiment)
    if experiment not in SUPPORTED_EXPERIMENTS:
        known = ", ".join(SUPPORTED_EXPERIMENTS)
        raise ValueError(
            f"Unknown terminal-reward experiment {experiment!r}; "
            f"expected one of: {known}."
        )
    return experiment


def final_evaluation_output_root(
    project_root: Path,
    agent_directory: str,
    map_name: str,
    agent_name: str,
) -> Path:
    return (
        Path(project_root)
        / "Agent Performance Charts"
        / agent_directory
        / map_name
        / agent_name
        / "final_evaluation"
    )


def _manifest_total_timesteps(save_root: Path) -> int:
    manifest_path = Path(save_root) / "latest_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Seed manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    total_timesteps = int(manifest.get("total_timesteps", 0))
    if total_timesteps < 1:
        raise RuntimeError(
            f"Manifest {manifest_path} has no valid total_timesteps value."
        )
    return total_timesteps


def _pt_final_steps(checkpoint_path: Path) -> int:
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return int(checkpoint.get("env_steps", 0))


def _zip_final_steps(checkpoint_path: Path) -> int:
    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        if "data" not in archive.namelist():
            return 0
        payload = json.loads(archive.read("data"))
    return int(payload.get("num_timesteps", 0))


def collect_final_checkpoint(
    save_root: Path,
    agent_name: str,
    seed: int,
    suffix: str,
):
    """
    Return only the completed seed's exact ``*_final`` checkpoint.

    The return shape matches the established checkpoint-sweep collectors:
    ``[(actual_training_steps, checkpoint_path)]``.
    """

    suffix = str(suffix).lower()
    if suffix not in {".pt", ".zip"}:
        raise ValueError(f"Unsupported final-checkpoint suffix: {suffix!r}")

    seed_dir = Path(save_root) / f"seed_{int(seed)}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    checkpoint_path = seed_dir / f"{agent_name}_final{suffix}"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Final checkpoint not found. Final-only evaluation requires a "
            f"completed training seed: {checkpoint_path}"
        )

    if suffix == ".pt":
        checkpoint_steps = _pt_final_steps(checkpoint_path)
    else:
        checkpoint_steps = _zip_final_steps(checkpoint_path)
    if checkpoint_steps < 1:
        checkpoint_steps = _manifest_total_timesteps(Path(save_root))

    return [(int(checkpoint_steps), checkpoint_path)]
