import argparse
import atexit
import collections
import importlib
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Agents.checkpoint_sweep_eval_common import (
    build_record,
    canonical_outcome_label,
    load_existing_results,
    load_latest_seed_manifest,
    normalize_results_df,
    plot_best_seed_stacked,
    plot_combined,
    save_results_csv,
    write_metadata,
)


MAP_VARIANTS = (
    "V1_Base",
    "V1_Combat",
    "V1_Navigate",
    "V2_Base",
    "V2_Combat",
    "V2_Navigate",
    "V3_Base",
    "V3_Combat",
    "V3_Navigate",
)
DEFAULT_AGENT_NAME = "MaskPPO_NS_reduced"
DEFAULT_EVAL_EPISODES = 10
DEFAULT_NUM_EVAL_ENVS = 5


class FlattenActionWrapper(Wrapper):
    """
    Keeps the NS env's per-friendly MultiDiscrete action space and flattens
    the 2D action mask so MaskablePPO can consume it.
    """

    def __init__(self, env, n_friend):
        super().__init__(env)

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise TypeError("Expected MultiDiscrete action_space from NS env")
        self.action_space = env.action_space
        self._n_agents = int(len(self.action_space.nvec))
        if self._n_agents != int(n_friend):
            raise ValueError(
                f"Unexpected action_space rank {self._n_agents}; expected N_FRIEND={n_friend}"
            )
        if np.unique(self.action_space.nvec).size != 1:
            raise ValueError("Expected identical per-friendly action dimensions")
        self._n_unit_actions = int(self.action_space.nvec[0])

        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)
        self._last_mask = np.ones(flat_len, dtype=np.int8)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        obs, rew, term, trunc, info = self.env.step(action_arr)
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        am = np.asarray(obs["action_mask"], dtype=np.int8)
        expected_shape = (self._n_agents, self._n_unit_actions)
        if am.shape != expected_shape:
            raise ValueError(f"action_mask shape {am.shape} != expected {expected_shape}")

        flat_mask = am.reshape(-1).astype(np.int8, copy=True)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask


def env_module_name(map_name):
    return f"Environments.NS_AM_RM_mean_reduced.{map_name}_NS"


def parse_args(default_map_name=None, default_agent_name=DEFAULT_AGENT_NAME):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved MaskPPO reduced checkpoint for each seed and plot "
            "win rate vs timesteps."
        )
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default=default_map_name,
        choices=MAP_VARIANTS,
        required=default_map_name is None,
        help="Map variant to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument(
        "--num-eval-envs",
        type=int,
        default=DEFAULT_NUM_EVAL_ENVS,
        help="Number of parallel SC2 eval environments to run per checkpoint.",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default=default_agent_name,
        help=(
            "Saved-model folder and checkpoint prefix to evaluate, e.g. "
            "MaskPPO_NS_reduced or MaskPPO_NS_reduced_pathable_only."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing cached CSV for this episode count and re-run all checkpoints.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic prediction.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return device_name


def validate_args(args):
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1.")
    if args.num_eval_envs < 1:
        raise ValueError("--num-eval-envs must be at least 1.")


def get_agent_save_root(map_name, agent_name):
    return (
        PROJECT_ROOT
        / "Agents"
        / "MaskPPO"
        / map_name
        / "saved_models"
        / agent_name
    )


def get_output_root(map_name, agent_name):
    return (
        PROJECT_ROOT
        / "Agent Performance Charts"
        / "MaskPPO"
        / map_name
        / agent_name
        / "checkpoint_sweep"
    )


def load_resume_results(output_root, primary_csv_path, episodes, mode_tag, overwrite):
    if overwrite:
        return pd.DataFrame()

    csv_paths = [primary_csv_path]
    pattern = f"checkpoint_metrics_{int(episodes)}ep_{mode_tag}_nenv*.csv"
    if output_root.is_dir():
        for csv_path in sorted(output_root.glob(pattern)):
            if csv_path not in csv_paths:
                csv_paths.append(csv_path)

    frames = []
    loaded_paths = []
    for csv_path in csv_paths:
        results_df = load_existing_results(csv_path, overwrite=False)
        if results_df.empty:
            continue
        if "episodes" in results_df.columns:
            results_df = results_df[results_df["episodes"].astype(int) == int(episodes)]
        if results_df.empty:
            continue
        frames.append(results_df)
        loaded_paths.append(csv_path)

    if not frames:
        return pd.DataFrame()

    results_df = normalize_results_df(pd.concat(frames, ignore_index=True))
    results_df = results_df.drop_duplicates(
        subset=["seed", "checkpoint_steps"],
        keep="last",
    )
    if loaded_paths:
        print("Resume cache loaded from:")
        for csv_path in loaded_paths:
            print(f"  {csv_path}")
    return normalize_results_df(results_df)


def parse_checkpoint_steps(agent_name, checkpoint_path):
    checkpoint_name = Path(checkpoint_path).name
    if "_final" in Path(checkpoint_name).stem:
        return None

    match = re.fullmatch(
        rf"{re.escape(agent_name)}_(\d+)([KMB]?)\.zip",
        checkpoint_name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    value = int(match.group(1))
    suffix = match.group(2).upper()
    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers[suffix]


def collect_seed_checkpoints(save_root, agent_name, seed):
    seed_dir = save_root / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    checkpoints = []
    for entry in seed_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".zip":
            continue
        checkpoint_steps = parse_checkpoint_steps(agent_name, entry)
        if checkpoint_steps is None:
            continue
        checkpoints.append((checkpoint_steps, entry))

    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    if not checkpoints:
        raise RuntimeError(f"No step checkpoints found under {seed_dir}")
    return checkpoints


def make_env(rank, base_seed, map_name, include_player_relative):
    def _init():
        worker_seed = int(base_seed) + int(rank)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        # Give each worker a private temp root to avoid Windows SC2 collisions.
        worker_tmp_dir = tempfile.mkdtemp(prefix=f"tbm-eval-worker-{rank}-")
        atexit.register(lambda path=worker_tmp_dir: shutil.rmtree(path, ignore_errors=True))
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = worker_tmp_dir

        # Stagger startup slightly so SC2 ports/temp files settle.
        time.sleep(0.5 * rank)

        env_module = importlib.import_module(env_module_name(map_name))
        env = env_module.TwoBridgeEnv(
            visualize=False,
            realtime=False,
            include_player_relative=bool(include_player_relative),
        )
        return FlattenActionWrapper(env, n_friend=env_module.N_FRIEND)

    return _init


def create_vec_env(num_envs, seed, map_name, include_player_relative):
    env_fns = [
        make_env(
            rank=rank,
            base_seed=seed,
            map_name=map_name,
            include_player_relative=include_player_relative,
        )
        for rank in range(num_envs)
    ]
    return SubprocVecEnv(env_fns, start_method="spawn")


def get_vec_action_masks(vec_env):
    masks = vec_env.env_method("action_masks")
    return np.asarray(masks, dtype=np.int8)


def evaluate_checkpoint(vec_env, checkpoint_path, eval_episodes, deterministic, device, seed):
    model = MaskablePPO.load(
        str(checkpoint_path),
        env=vec_env,
        device=device,
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )
    raw_counts = collections.Counter()
    episodes_assigned = 0
    episodes_finished = 0
    num_eval_envs = int(vec_env.num_envs)

    while episodes_finished < eval_episodes:
        batch_size = min(num_eval_envs, eval_episodes - episodes_assigned)
        if batch_size <= 0:
            break

        vec_env.seed(int(seed) + episodes_assigned)
        obs = vec_env.reset()
        episodes_assigned += batch_size

        active_envs = np.zeros(num_eval_envs, dtype=bool)
        active_envs[:batch_size] = True
        completed_envs = ~active_envs.copy()

        while not np.all(completed_envs):
            action_masks = get_vec_action_masks(vec_env)
            actions, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            actions = np.asarray(actions, dtype=np.int64)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            actions[completed_envs] = 0

            obs, _, dones, infos = vec_env.step(actions)
            for env_idx, done in enumerate(dones):
                if not active_envs[env_idx] or completed_envs[env_idx] or not done:
                    continue
                raw_result = canonical_outcome_label(infos[env_idx].get("result"))
                raw_counts[raw_result] += 1
                completed_envs[env_idx] = True
                episodes_finished += 1

    if episodes_finished != eval_episodes:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} finished {episodes_finished} eval episodes, "
            f"expected {eval_episodes}."
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_counts


def resolve_include_player_relative(manifest, agent_name):
    if "include_player_relative" in manifest:
        return bool(manifest["include_player_relative"])
    return not str(agent_name).endswith("_pathable_only")


def main(default_map_name=None, default_agent_name=DEFAULT_AGENT_NAME):
    args = parse_args(default_map_name, default_agent_name)
    validate_args(args)
    map_name = args.map_name
    deterministic = not args.stochastic
    device = resolve_device(args.device)
    mode_tag = "det" if deterministic else "stoch"
    run_tag = f"{args.episodes}ep_{mode_tag}_nenv{args.num_eval_envs}"

    save_root = get_agent_save_root(map_name, args.agent_name)
    manifest_path, manifest = load_latest_seed_manifest(save_root)
    seeds = manifest["seeds"]
    include_player_relative = resolve_include_player_relative(manifest, args.agent_name)
    output_root = get_output_root(map_name, args.agent_name)
    csv_path = output_root / f"checkpoint_metrics_{run_tag}.csv"
    metadata_path = output_root / f"checkpoint_eval_metadata_{run_tag}.json"
    combined_plot_path = output_root / f"all_seeds_winrate_vs_timesteps_{run_tag}.png"
    best_seed_plot_path = output_root / f"best_seed_win_conditions_vs_timesteps_{run_tag}.png"

    results_df = load_resume_results(
        output_root=output_root,
        primary_csv_path=csv_path,
        episodes=args.episodes,
        mode_tag=mode_tag,
        overwrite=args.overwrite,
    )
    existing_keys = {
        (int(row.seed), int(row.checkpoint_steps))
        for row in results_df.itertuples(index=False)
    }
    all_records = results_df.to_dict("records")

    print(f"Agent: {args.agent_name}")
    print(f"Map: {map_name}")
    print(f"Env module: {env_module_name(map_name)}")
    print(f"Device: {device}")
    print(f"Episodes per checkpoint: {args.episodes}")
    print(f"Parallel eval envs per checkpoint: {args.num_eval_envs}")
    print(f"Deterministic policy: {deterministic}")
    print(f"include_player_relative: {include_player_relative}")
    print(f"Manifest: {manifest_path}")
    print(f"Seeds: {list(seeds)}")
    print(f"Cached rows found: {len(existing_keys)}")

    for seed_idx, seed in enumerate(seeds, start=1):
        checkpoints = collect_seed_checkpoints(save_root, args.agent_name, seed)
        cached_for_seed = {
            int(checkpoint_steps)
            for cached_seed, checkpoint_steps in existing_keys
            if int(cached_seed) == int(seed)
        }
        checkpoint_steps_for_seed = {
            int(checkpoint_steps) for checkpoint_steps, _checkpoint_path in checkpoints
        }
        print(
            f"\nSeed {seed_idx}/{len(seeds)} | seed_{seed} | "
            f"checkpoint_count={len(checkpoints)}"
        )
        if checkpoint_steps_for_seed and checkpoint_steps_for_seed.issubset(cached_for_seed):
            print(
                f"  seed cached complete | skip env creation | "
                f"cached_checkpoints={len(checkpoint_steps_for_seed)}"
            )
            continue

        env = create_vec_env(
            num_envs=args.num_eval_envs,
            seed=int(seed) + 10_000,
            map_name=map_name,
            include_player_relative=include_player_relative,
        )
        try:
            env.seed(int(seed) + 10_000)
            obs = env.reset()
            print(
                "  Eval env ready | "
                f"num_envs={env.num_envs} | "
                f"minimap={obs['minimap'].shape} | "
                f"vector={obs['vector'].shape} | "
                f"action_mask={obs['action_mask'].shape}"
            )
            for checkpoint_idx, (checkpoint_steps, checkpoint_path) in enumerate(
                checkpoints, start=1
            ):
                key = (int(seed), int(checkpoint_steps))
                if key in existing_keys:
                    print(
                        f"  [{checkpoint_idx}/{len(checkpoints)}] "
                        f"skip cached | step={checkpoint_steps} | file={checkpoint_path.name}"
                    )
                    continue

                print(
                    f"  [{checkpoint_idx}/{len(checkpoints)}] "
                    f"eval start | step={checkpoint_steps} | file={checkpoint_path.name}"
                )
                raw_counts = evaluate_checkpoint(
                    vec_env=env,
                    checkpoint_path=checkpoint_path,
                    eval_episodes=args.episodes,
                    deterministic=deterministic,
                    device=device,
                    seed=int(seed) + int(checkpoint_steps),
                )
                record = build_record(
                    seed=seed,
                    checkpoint_steps=checkpoint_steps,
                    checkpoint_path=checkpoint_path,
                    eval_episodes=args.episodes,
                    raw_counts=raw_counts,
                )
                all_records.append(record)
                existing_keys.add(key)
                results_df = normalize_results_df(pd.DataFrame(all_records))
                save_results_csv(results_df, csv_path)
                print(
                    f"  [{checkpoint_idx}/{len(checkpoints)}] "
                    f"eval done | step={checkpoint_steps} | "
                    f"win_rate={record['win_rate_percent']:.2f}% | "
                    f"nav_win={record['nav_win']} | combat_win={record['combat_win']} | "
                    f"combat_loss={record['combat_loss']} | nav_loss={record['nav_loss']} | "
                    f"unexpected={record['unexpected_count']}"
                )
        finally:
            env.close()

    results_df = normalize_results_df(pd.DataFrame(all_records))
    save_results_csv(results_df, csv_path)
    if results_df.empty:
        raise RuntimeError("No checkpoint evaluation rows were produced.")

    plot_combined(results_df, args.agent_name, map_name, combined_plot_path)
    best_seed = plot_best_seed_stacked(
        results_df,
        args.agent_name,
        map_name,
        best_seed_plot_path,
    )
    write_metadata(
        metadata_path,
        args.agent_name,
        map_name,
        manifest_path,
        manifest,
        args,
        device,
        csv_path,
        results_df,
        combined_plot_path,
        best_seed_plot_path,
        best_seed,
    )

    print("\nOutputs")
    print(f"  Results CSV: {csv_path}")
    print(f"  Metadata JSON: {metadata_path}")
    print(f"  Combined plot: {combined_plot_path}")
    print(f"  Best seed stacked plot: {best_seed_plot_path}")
    print(f"  Best seed: {best_seed}")


def main_for_script(script_path, default_agent_name=DEFAULT_AGENT_NAME):
    main(
        default_map_name=Path(script_path).resolve().parent.name,
        default_agent_name=default_agent_name,
    )


if __name__ == "__main__":
    main()
