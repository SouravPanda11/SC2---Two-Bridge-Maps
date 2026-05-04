import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Agents.checkpoint_sweep_eval_common import (
    build_record,
    canonical_outcome_label,
    collect_seed_checkpoints,
    load_existing_results,
    load_latest_seed_manifest,
    normalize_results_df,
    plot_best_seed_stacked,
    plot_combined,
    save_results_csv,
    write_metadata,
)
from Agents.MAPPO_reduced._train_mappo_reduced import (
    MAPPOPolicy,
    ParallelEnvBatch,
    make_config,
    safe_torch_load,
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
DEFAULT_AGENT_NAME = "MAPPO_reduced"
DEFAULT_EVAL_EPISODES = 32
DEFAULT_NUM_EVAL_ENVS = 16


def parse_args(default_map_name=None, default_agent_name=DEFAULT_AGENT_NAME):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved MAPPO checkpoint for each seed and plot "
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
            "MAPPO_reduced or MAPPO_reduced_pathable_only."
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
        help="Sample from the MAPPO policy instead of greedy action selection.",
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
        / "MAPPO_reduced"
        / map_name
        / "saved_models"
        / agent_name
    )


def get_output_root(map_name, agent_name):
    return (
        PROJECT_ROOT
        / "Agent Performance Charts"
        / "MAPPO_reduced"
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


def config_from_checkpoint(checkpoint, device, default_map_name):
    checkpoint_config = checkpoint.get("config", {})
    map_name = checkpoint_config.get("map_name", default_map_name)
    config = make_config(map_name)
    for key, value in checkpoint_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.device = device
    config.eval_during_training = False
    config.use_tensorboard = False
    config.visualize = False
    config.realtime = False
    config.replay_dir = ""
    config.save_replay_episodes = 0
    config.resume_checkpoint = ""
    return config


class MAPPOEvalPolicy:
    def __init__(self, checkpoint_path, env_info, device, map_name):
        self.device = device
        checkpoint = safe_torch_load(Path(checkpoint_path), map_location=device)
        self.config = config_from_checkpoint(checkpoint, device, map_name)

        self.n_agents = int(env_info["n_agents"])
        self.n_actions = int(env_info["n_actions"])
        self.obs_dim = int(env_info["obs_shape"])
        self.state_dim = int(env_info["state_shape"])
        self.minimap_shape = tuple(int(dim) for dim in env_info["minimap_shape"])

        self.policy = MAPPOPolicy(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            minimap_shape=self.minimap_shape,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            config=self.config,
        ).to(self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()

    @torch.no_grad()
    def act(self, obs_batch, state_batch, minimap_batch, avail_actions_batch, greedy):
        actions, _, _ = self.policy.act(
            obs_batch,
            state_batch,
            minimap_batch,
            avail_actions_batch,
            greedy=greedy,
        )
        return actions


def create_eval_env(map_name, num_envs, seed, include_player_relative):
    env_kwargs = {
        "map_name": map_name,
        "episode_limit": None,
        "visualize": False,
        "realtime": False,
        "replay_dir": "",
        "save_replay_episodes": 0,
        "include_player_relative": bool(include_player_relative),
    }
    return ParallelEnvBatch(
        num_envs=int(num_envs),
        base_seed=int(seed),
        map_name=map_name,
        env_kwargs=env_kwargs,
    )


def evaluate_checkpoint(env, checkpoint_path, eval_episodes, device, seed, greedy, map_name):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    policy = MAPPOEvalPolicy(
        checkpoint_path=checkpoint_path,
        env_info=env.get_env_info(),
        device=device,
        map_name=map_name,
    )
    raw_counts = collections.Counter()
    num_envs = int(env.num_envs)
    n_agents = int(policy.n_agents)
    obs_dim = int(policy.obs_dim)
    state_dim = int(policy.state_dim)
    minimap_shape = tuple(policy.minimap_shape)
    n_actions = int(policy.n_actions)

    current = {
        "obs": np.zeros((num_envs, n_agents, obs_dim), dtype=np.float32),
        "state": np.zeros((num_envs, state_dim), dtype=np.float32),
        "minimap": np.zeros((num_envs, *minimap_shape), dtype=np.uint8),
        "avail_actions": np.zeros((num_envs, n_agents, n_actions), dtype=np.float32),
    }
    active = np.zeros(num_envs, dtype=bool)
    episodes_started = 0
    episodes_finished = 0

    while episodes_finished < eval_episodes:
        free_indices = [idx for idx in range(num_envs) if not active[idx]]
        start_count = min(len(free_indices), eval_episodes - episodes_started)
        if start_count > 0:
            start_indices = free_indices[:start_count]
            for env_idx, payload in env.reset(start_indices):
                current["obs"][env_idx] = payload["obs"]
                current["state"][env_idx] = payload["state"]
                current["minimap"][env_idx] = payload["minimap"]
                current["avail_actions"][env_idx] = payload["avail_actions"]
                active[env_idx] = True
            episodes_started += start_count

        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            continue
        active_list = active_indices.tolist()

        actions = np.zeros((num_envs, n_agents), dtype=np.int64)
        selected_actions = policy.act(
            obs_batch=current["obs"][active_list],
            state_batch=current["state"][active_list],
            minimap_batch=current["minimap"][active_list],
            avail_actions_batch=current["avail_actions"][active_list],
            greedy=greedy,
        )
        for local_idx, env_idx in enumerate(active_list):
            actions[env_idx] = selected_actions[local_idx]

        for env_idx, payload in env.step(actions, active_list):
            done = bool(payload["terminated"] or payload["truncated"])
            if done:
                raw_result = canonical_outcome_label(payload.get("info", {}).get("result"))
                raw_counts[raw_result] += 1
                active[env_idx] = False
                episodes_finished += 1
                continue

            current["obs"][env_idx] = payload["obs"]
            current["state"][env_idx] = payload["state"]
            current["minimap"][env_idx] = payload["minimap"]
            current["avail_actions"][env_idx] = payload["avail_actions"]

    if episodes_finished != eval_episodes:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} finished {episodes_finished} eval episodes, "
            f"expected {eval_episodes}."
        )
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_counts


def main(default_map_name=None, default_agent_name=DEFAULT_AGENT_NAME):
    args = parse_args(default_map_name, default_agent_name)
    validate_args(args)
    map_name = args.map_name
    device = resolve_device(args.device)
    greedy = not args.stochastic
    mode_tag = "greedy" if greedy else "stoch"
    run_tag = f"{args.episodes}ep_{mode_tag}_nenv{args.num_eval_envs}"

    save_root = get_agent_save_root(map_name, args.agent_name)
    manifest_path, manifest = load_latest_seed_manifest(save_root)
    seeds = manifest["seeds"]
    include_player_relative = bool(
        manifest.get("include_player_relative", not args.agent_name.endswith("_pathable_only"))
    )
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
    print(f"Device: {device}")
    print(f"Episodes per checkpoint: {args.episodes}")
    print(f"Parallel eval envs per checkpoint: {args.num_eval_envs}")
    print(f"Greedy policy: {greedy}")
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

        env = create_eval_env(
            map_name=map_name,
            num_envs=args.num_eval_envs,
            seed=int(seed) + 10_000,
            include_player_relative=include_player_relative,
        )
        try:
            print(f"  Eval env ready | num_envs={env.num_envs} | env_info={env.get_env_info()}")
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
                    env=env,
                    checkpoint_path=checkpoint_path,
                    eval_episodes=args.episodes,
                    device=device,
                    seed=int(seed) + int(checkpoint_steps),
                    greedy=greedy,
                    map_name=map_name,
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
    import multiprocessing as mp

    mp.freeze_support()
    main()
