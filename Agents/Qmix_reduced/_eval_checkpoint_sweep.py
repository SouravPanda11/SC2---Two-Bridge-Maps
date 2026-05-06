import argparse
import collections
import importlib
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
DEFAULT_AGENT_NAME = "QMIX_reduced"
DEFAULT_EVAL_EPISODES = 32
DEFAULT_NUM_EVAL_ENVS = 16


def parse_args(default_map_name=None, default_agent_name=DEFAULT_AGENT_NAME):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved QMIX_reduced checkpoint for each seed and plot "
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
            "QMIX_reduced or QMIX_reduced_pathable_only."
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
        help="Use epsilon-greedy action sampling instead of greedy action selection.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Exploration probability used only with --stochastic.",
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
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("--epsilon must be between 0 and 1.")


def load_trainer_module(map_name):
    return importlib.import_module(f"Agents.Qmix_reduced.{map_name}.train_qmix")


def get_agent_save_root(map_name, agent_name):
    return (
        PROJECT_ROOT
        / "Agents"
        / "Qmix_reduced"
        / map_name
        / "saved_models"
        / agent_name
    )


def get_output_root(map_name, agent_name):
    return (
        PROJECT_ROOT
        / "Agent Performance Charts"
        / "Qmix_reduced"
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


def config_from_checkpoint(trainer_module, checkpoint, device):
    config = trainer_module.TrainConfig()
    for key, value in checkpoint.get("config", {}).items():
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


class QMixReducedEvalPolicy:
    def __init__(self, trainer_module, checkpoint_path, env_info, device):
        self.trainer_module = trainer_module
        self.device = device
        checkpoint = trainer_module.safe_torch_load(Path(checkpoint_path), map_location=device)
        self.config = config_from_checkpoint(trainer_module, checkpoint, device)

        self.n_agents = int(env_info["n_agents"])
        self.n_actions = int(env_info["n_actions"])
        self.obs_dim = int(env_info["obs_shape"])
        self.minimap_shape = tuple(int(dim) for dim in env_info.get("minimap_shape", ()))
        self.use_minimap = bool(self.config.use_minimap and self.minimap_shape)

        self.minimap_encoder = None
        self.minimap_embed_dim = 0
        if self.use_minimap:
            self.minimap_embed_dim = int(self.config.minimap_embed_dim)
            self.minimap_encoder = trainer_module.MinimapEncoder(
                in_channels=self.minimap_shape[0],
                height=self.minimap_shape[1],
                width=self.minimap_shape[2],
                embed_dim=self.minimap_embed_dim,
            ).to(self.device)
            self.minimap_encoder.load_state_dict(checkpoint["minimap_encoder_state_dict"])
            self.minimap_encoder.eval()

        agent_input_dim = (
            self.obs_dim
            + self.minimap_embed_dim
            + (self.n_agents if self.config.obs_agent_id else 0)
        )
        self.agent = trainer_module.SharedQAgent(
            input_dim=agent_input_dim,
            hidden_dim=self.config.hidden_dim,
            n_actions=self.n_actions,
            use_rnn=self.config.use_rnn,
        ).to(self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.agent.eval()

    def init_hidden(self, num_envs):
        hidden = self.agent.init_hidden(int(num_envs) * self.n_agents, self.device)
        return hidden.view(int(num_envs), self.n_agents, -1)

    def encode_minimap(self, minimap_batch):
        if self.minimap_encoder is None:
            return None
        minimap_tensor = torch.as_tensor(minimap_batch, dtype=torch.uint8, device=self.device)
        return self.minimap_encoder(minimap_tensor)

    def build_agent_inputs(self, obs_t, minimap_embed_t=None):
        batch_size = obs_t.size(0)
        pieces = [obs_t]
        if minimap_embed_t is not None:
            minimap_features = minimap_embed_t.unsqueeze(1).expand(-1, self.n_agents, -1)
            pieces.append(minimap_features)
        if self.config.obs_agent_id:
            agent_ids = (
                torch.eye(self.n_agents, device=obs_t.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            pieces.append(agent_ids)
        return torch.cat(pieces, dim=-1).reshape(batch_size * self.n_agents, -1)

    @torch.no_grad()
    def select_actions_batch(
        self,
        obs_batch,
        minimap_batch,
        avail_actions_batch,
        hidden_batch,
        greedy,
        epsilon,
    ):
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        avail_tensor = torch.as_tensor(avail_actions_batch, dtype=torch.bool, device=self.device)
        minimap_embed = self.encode_minimap(minimap_batch)
        inputs = self.build_agent_inputs(obs_tensor, minimap_embed)

        batch_size = obs_tensor.size(0)
        hidden_flat = hidden_batch.reshape(batch_size * self.n_agents, -1)
        q_values, next_hidden = self.agent(inputs, hidden_flat)
        q_values = q_values.view(batch_size, self.n_agents, self.n_actions)
        next_hidden = next_hidden.view(batch_size, self.n_agents, -1)

        masked_q = q_values.clone()
        masked_q[~avail_tensor] = -1e9
        greedy_actions = masked_q.argmax(dim=2)
        if greedy:
            return greedy_actions.cpu().numpy().astype(np.int64), next_hidden.detach()

        chosen = np.zeros((batch_size, self.n_agents), dtype=np.int64)
        for env_idx in range(batch_size):
            for agent_id in range(self.n_agents):
                valid_actions = torch.nonzero(
                    avail_tensor[env_idx, agent_id], as_tuple=False
                ).squeeze(-1)
                if valid_actions.numel() == 0:
                    chosen[env_idx, agent_id] = 0
                elif random.random() < epsilon:
                    chosen[env_idx, agent_id] = int(
                        valid_actions[random.randrange(valid_actions.numel())].item()
                    )
                else:
                    chosen[env_idx, agent_id] = int(greedy_actions[env_idx, agent_id].item())
        return chosen, next_hidden.detach()


def create_eval_env(trainer_module, num_envs, seed, include_player_relative):
    env_kwargs = {
        "map_name": trainer_module.MAP_NAME,
        "episode_limit": None,
        "visualize": False,
        "realtime": False,
        "replay_dir": "",
        "save_replay_episodes": 0,
        "include_player_relative": bool(include_player_relative),
    }
    return trainer_module.ParallelQMixEnvBatch(
        num_envs=int(num_envs),
        base_seed=int(seed),
        env_kwargs=env_kwargs,
    )


def evaluate_checkpoint(
    trainer_module,
    env,
    checkpoint_path,
    eval_episodes,
    device,
    seed,
    greedy,
    epsilon,
):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    policy = QMixReducedEvalPolicy(
        trainer_module=trainer_module,
        checkpoint_path=checkpoint_path,
        env_info=env.get_env_info(),
        device=device,
    )
    raw_counts = collections.Counter()
    num_envs = int(env.num_envs)
    n_agents = int(policy.n_agents)
    obs_dim = int(policy.obs_dim)
    minimap_shape = tuple(policy.minimap_shape)
    n_actions = int(policy.n_actions)

    current = {
        "obs": np.zeros((num_envs, n_agents, obs_dim), dtype=np.float32),
        "minimap": np.zeros((num_envs, *minimap_shape), dtype=np.uint8),
        "avail_actions": np.zeros((num_envs, n_agents, n_actions), dtype=np.float32),
    }
    hidden = policy.init_hidden(num_envs)
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
                current["minimap"][env_idx] = payload["minimap"]
                current["avail_actions"][env_idx] = payload["avail_actions"]
                hidden[env_idx] = policy.init_hidden(1)[0]
                active[env_idx] = True
            episodes_started += start_count

        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            continue
        active_list = active_indices.tolist()

        actions = np.zeros((num_envs, n_agents), dtype=np.int64)
        selected_actions, next_hidden = policy.select_actions_batch(
            obs_batch=current["obs"][active_list],
            minimap_batch=current["minimap"][active_list],
            avail_actions_batch=current["avail_actions"][active_list],
            hidden_batch=hidden[active_list],
            greedy=greedy,
            epsilon=epsilon,
        )
        for local_idx, env_idx in enumerate(active_list):
            actions[env_idx] = selected_actions[local_idx]
            hidden[env_idx] = next_hidden[local_idx]

        for env_idx, payload in env.step(actions, active_list):
            done = bool(payload["terminated"] or payload["truncated"])
            if done:
                raw_result = canonical_outcome_label(payload.get("info", {}).get("result"))
                raw_counts[raw_result] += 1
                active[env_idx] = False
                episodes_finished += 1
                continue

            current["obs"][env_idx] = payload["obs"]
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


def resolve_include_player_relative(save_root, manifest, seeds, agent_name):
    if "include_player_relative" in manifest:
        return bool(manifest["include_player_relative"])

    first_run_config = save_root / f"seed_{int(seeds[0])}" / "run_config.json"
    if first_run_config.is_file():
        import json

        with first_run_config.open("r", encoding="utf-8") as handle:
            return bool(json.load(handle).get("include_player_relative", True))

    return not str(agent_name).endswith("_pathable_only")


def main(default_map_name=None, default_agent_name=DEFAULT_AGENT_NAME):
    args = parse_args(default_map_name, default_agent_name)
    validate_args(args)
    map_name = args.map_name
    trainer_module = load_trainer_module(map_name)
    device = resolve_device(args.device)
    greedy = not args.stochastic
    epsilon = float(args.epsilon if args.stochastic else 0.0)
    mode_tag = "greedy" if greedy else f"eps{str(epsilon).replace('.', 'p')}"
    run_tag = f"{args.episodes}ep_{mode_tag}_nenv{args.num_eval_envs}"

    save_root = get_agent_save_root(map_name, args.agent_name)
    manifest_path, manifest = load_latest_seed_manifest(save_root)
    seeds = manifest["seeds"]
    include_player_relative = resolve_include_player_relative(
        save_root=save_root,
        manifest=manifest,
        seeds=seeds,
        agent_name=args.agent_name,
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
    print(f"Epsilon: {epsilon}")
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
            trainer_module=trainer_module,
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
                    trainer_module=trainer_module,
                    env=env,
                    checkpoint_path=checkpoint_path,
                    eval_episodes=args.episodes,
                    device=device,
                    seed=int(seed) + int(checkpoint_steps),
                    greedy=greedy,
                    epsilon=epsilon,
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
