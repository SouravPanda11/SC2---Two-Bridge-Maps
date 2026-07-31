"""Shared evaluator for the Two-Bridge scripted combat baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.scripted.policy import (  # noqa: E402
    ATTACK,
    MOVE,
    MOVE_DELTAS,
    AgentConfig,
    LowerBridgeCombatAgent,
    ScriptedCommand,
    UnitSnapshot,
    WorldSnapshot,
    distance,
)
from Agents.scripted.variants import (  # noqa: E402
    CANONICAL_VARIANT_NAMES,
    MAP_VARIANTS,
    MapVariant,
    get_variant,
)


DEFAULT_INSTALLED_MAP_DIR = Path(
    r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
)
DEFAULT_REPO_MAP_DIR = PROJECT_ROOT / "Maps" / "Camera Free"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Agent Performance Charts" / "Scripted"

BEACON_TYPE_ID = 317
BEACON_RADIUS = 5.0
DEFAULT_STEP_MUL = 8
DEFAULT_MAX_STEPS = (5 * 60 * 16) // DEFAULT_STEP_MUL


def _import_pysc2():
    try:
        from absl import flags
        from pysc2.env import sc2_env
        from pysc2.lib import actions
        from pysc2.maps import lib
    except Exception as exc:  # pragma: no cover - exercised only without runtime deps
        raise RuntimeError(
            "PySC2 is unavailable. Run with the project environment, for example "
            "`TBMsc2\\Scripts\\python.exe`."
        ) from exc

    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]])
    return sc2_env, actions, lib


def resolve_map_dir(requested: str | Path | None) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    if DEFAULT_INSTALLED_MAP_DIR.is_dir():
        return DEFAULT_INSTALLED_MAP_DIR
    return DEFAULT_REPO_MAP_DIR.resolve()


def resolve_variant_map_file(map_dir: Path, variant: MapVariant) -> Path:
    """Resolve a map name while preserving its on-disk capitalization."""

    expected = map_dir / variant.filename
    if expected.is_file():
        return expected
    if map_dir.is_dir():
        casefolded_name = variant.filename.casefold()
        matches = sorted(
            path
            for path in map_dir.iterdir()
            if path.is_file() and path.name.casefold() == casefolded_name
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing Two-Bridge map file: {expected}")


def validate_map_files(map_dir: Path, variants: Iterable[MapVariant]) -> None:
    missing = []
    for variant in variants:
        try:
            resolve_variant_map_file(map_dir, variant)
        except FileNotFoundError:
            missing.append(str(map_dir / variant.filename))
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing Two-Bridge map file(s):\n  - {formatted}")


def register_map(lib_module, variant: MapVariant, map_dir: Path) -> None:
    # The unique Scripted... registry name means this cannot collide with the
    # classes declared by the legacy Gym environment modules.
    if variant.registry_name in lib_module.get_maps():
        return
    map_file = resolve_variant_map_file(map_dir, variant)
    type(
        variant.registry_name,
        (lib_module.Map,),
        {
            "directory": str(map_file.parent),
            "filename": map_file.name,
            "players": 2,
        },
    )


def snapshot_from_timestep(timestep) -> WorldSnapshot:
    raw_units = tuple(timestep.observation.raw_units)

    def convert(unit) -> UnitSnapshot:
        return UnitSnapshot(
            tag=int(unit.tag),
            x=float(unit.x),
            y=float(unit.y),
            health=float(unit.health),
            weapon_cooldown=float(getattr(unit, "weapon_cooldown", 0.0)),
        )

    friends = tuple(
        sorted(
            (convert(unit) for unit in raw_units if unit.owner == 1 and unit.health > 0),
            key=lambda unit: unit.tag,
        )
    )
    enemies = tuple(
        sorted(
            (convert(unit) for unit in raw_units if unit.owner == 2 and unit.health > 0),
            key=lambda unit: unit.tag,
        )
    )
    beacon_unit = next(
        (unit for unit in raw_units if unit.unit_type == BEACON_TYPE_ID),
        None,
    )
    beacon = convert(beacon_unit) if beacon_unit is not None else None
    game_loop_value = timestep.observation.game_loop
    try:
        game_loop = int(game_loop_value[0])
    except (IndexError, TypeError):
        game_loop = int(game_loop_value)
    return WorldSnapshot(
        friends=friends,
        enemies=enemies,
        beacon=beacon,
        game_loop=game_loop,
    )


def infer_region(point: tuple[float, float]) -> str:
    """Return the nearest labeled spawn region R1-R6."""

    side_offset = 0 if point[0] < 28.0 else 3
    if point[1] >= 47.0:
        vertical_index = 1
    elif point[1] >= 20.0:
        vertical_index = 2
    else:
        vertical_index = 3
    return f"R{vertical_index + side_offset}"


def representative_region(units: tuple[UnitSnapshot, ...]) -> str | None:
    if not units:
        return None
    x = sum(unit.x for unit in units) / len(units)
    y = sum(unit.y for unit in units) / len(units)
    return infer_region((x, y))


def unit_centroid(units: tuple[UnitSnapshot, ...]) -> list[float] | None:
    if not units:
        return None
    return [
        sum(unit.x for unit in units) / len(units),
        sum(unit.y for unit in units) / len(units),
    ]


def classify_outcome(
    timestep,
    snapshot: WorldSnapshot,
    episode_step: int,
    max_steps: int,
) -> str | None:
    # Match the benchmark wrappers' terminal priority exactly: native SC2
    # terminal, beacon, unit-count outcomes, then the scripted time limit.
    if timestep.last():
        reward = float(timestep.reward)
        if reward > 0:
            # Existing repository evaluators normalize PySC2 victory this way.
            return "combat_win"
        if reward < 0:
            return "combat_loss"
        return "tie"

    no_friends = not snapshot.friends
    no_enemies = not snapshot.enemies
    if snapshot.beacon is not None and snapshot.friends:
        if min(
            distance(friend.position, snapshot.beacon.position)
            for friend in snapshot.friends
        ) < BEACON_RADIUS:
            return "nav_win"

    if no_friends and no_enemies:
        return "tie"
    if no_enemies:
        return "combat_win"
    if no_friends:
        return "combat_loss"
    if episode_step >= max_steps:
        return "timeout_loss"
    return None


def raw_actions_for_command(
    raw_functions,
    command: ScriptedCommand,
    snapshot: WorldSnapshot,
    map_size: tuple[float, float],
) -> list[Any]:
    friend_by_tag = {friend.tag: friend for friend in snapshot.friends}
    if command.verb == MOVE:
        delta = MOVE_DELTAS.get(command.direction)
        if delta is None:
            return []
        selected = tuple(
            friend_by_tag[tag]
            for tag in command.unit_tags
            if tag in friend_by_tag
        )
        if len(selected) > 1:
            # Preserve the requested "all Marines selected" route semantics.
            # A grouped Move_pt also lets SC2's formation/pathing system pass the
            # five-unit squad through the narrow bridge without body blocking.
            center_x = sum(friend.x for friend in selected) / len(selected)
            center_y = sum(friend.y for friend in selected) / len(selected)
            x = min(max(center_x + delta[0], 0.0), map_size[0] - 1.0)
            y = min(max(center_y + delta[1], 0.0), map_size[1] - 1.0)
            return [
                raw_functions.Move_pt(
                    "now",
                    [friend.tag for friend in selected],
                    (x, y),
                )
            ]
        raw_actions = []
        for friend in selected:
            x = min(max(friend.x + delta[0], 0.0), map_size[0] - 1.0)
            y = min(max(friend.y + delta[1], 0.0), map_size[1] - 1.0)
            raw_actions.append(raw_functions.Move_pt("now", [friend.tag], (x, y)))
        return raw_actions
    if command.verb == ATTACK:
        live_enemy_tags = {enemy.tag for enemy in snapshot.enemies}
        if (
            not command.unit_tags
            or command.target_tag is None
            or command.target_tag not in live_enemy_tags
        ):
            return []
        selected_tags = [
            tag for tag in command.unit_tags if tag in friend_by_tag
        ]
        if not selected_tags:
            return []
        return [
            raw_functions.Attack_unit(
                "now",
                selected_tags,
                int(command.target_tag),
            )
        ]
    return []


def _difficulty(sc2_env_module, name: str):
    try:
        return getattr(sc2_env_module.Difficulty, name)
    except AttributeError as exc:
        choices = ", ".join(item.name for item in sc2_env_module.Difficulty)
        raise ValueError(f"Unknown SC2 bot difficulty {name!r}. Choices: {choices}") from exc


def make_environment(
    variant: MapVariant,
    map_dir: Path,
    *,
    seed: int,
    step_mul: int,
    max_steps: int,
    bot_difficulty: str,
    visualize: bool,
    realtime: bool,
    replay_dir: Path | None,
):
    sc2_env, actions, lib = _import_pysc2()
    register_map(lib, variant, map_dir)
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    env = sc2_env.SC2Env(
        map_name=variant.registry_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, _difficulty(sc2_env, bot_difficulty)),
        ],
        step_mul=step_mul,
        game_steps_per_episode=max_steps * step_mul,
        random_seed=seed,
        disable_fog=True,
        ensure_available_actions=False,
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
            raw_crop_to_playable_area=True,
        ),
        visualize=visualize,
        realtime=realtime,
        replay_dir=str(replay_dir) if replay_dir is not None else None,
        replay_prefix=f"scripted_{variant.name}",
        save_replay_episodes=1 if replay_dir is not None else 0,
    )
    map_info = env.game_info[0].start_raw.map_size
    return env, actions.RAW_FUNCTIONS, (float(map_info.x), float(map_info.y))


def _initial_layout(snapshot: WorldSnapshot) -> dict[str, str | None]:
    return {
        "friendly_region": representative_region(snapshot.friends),
        "enemy_region": representative_region(snapshot.enemies),
        "beacon_region": (
            infer_region(snapshot.beacon.position) if snapshot.beacon is not None else None
        ),
    }


def evaluate_episode(
    env,
    raw_functions,
    map_size: tuple[float, float],
    *,
    variant: MapVariant,
    episode_index: int,
    episode_seed: int,
    agent_config: AgentConfig,
    max_steps: int,
) -> dict[str, Any]:
    # SC2Env reuses its configured seed on every restart. Advance it explicitly
    # so episodes are both reproducible and drawn from different trigger layouts.
    env._random_seed = int(episode_seed)
    timestep = env.reset()[0]
    snapshot = snapshot_from_timestep(timestep)
    controller = LowerBridgeCombatAgent(agent_config)
    controller.reset(snapshot)

    initial_friends = len(snapshot.friends)
    initial_enemies = len(snapshot.enemies)
    initial_friend_health = sum(unit.health for unit in snapshot.friends)
    initial_enemy_health = sum(unit.health for unit in snapshot.enemies)
    layout = _initial_layout(snapshot)
    episode_step = 0
    total_sc2_reward = float(timestep.reward)
    action_counts: Counter[str] = Counter()
    first_bridge_step: int | None = None
    first_contact_step: int | None = None
    first_damage_step: int | None = None
    first_attack_step: int | None = None
    started_at = time.perf_counter()

    while True:
        outcome = classify_outcome(timestep, snapshot, episode_step, max_steps)
        if outcome is not None:
            break

        commands = controller.act_many(snapshot)
        for command in commands:
            if command.verb == MOVE:
                action_counts["move"] += 1
            elif command.verb == ATTACK:
                action_counts["attack"] += 1
                if first_attack_step is None:
                    first_attack_step = episode_step
            else:
                action_counts["noop"] += 1

        # SC2Env expects an outer list per agent and an inner list whenever the
        # combat tactic emits simultaneous per-unit primitives. Several
        # repository wrappers omit this nesting, silently dropping later units.
        raw_actions = [
            raw_action
            for command in commands
            for raw_action in raw_actions_for_command(
                raw_functions,
                command,
                snapshot,
                map_size,
            )
        ]
        if not raw_actions:
            raw_actions = [raw_functions.no_op()]
        timestep = env.step([raw_actions])[0]
        episode_step += 1
        total_sc2_reward += float(timestep.reward)
        snapshot = snapshot_from_timestep(timestep)

        # Observe state changes immediately after the action as well as on the
        # next policy call, so logged milestone steps are exact.
        controller.observe(snapshot)
        if controller.bridge_reached and first_bridge_step is None:
            first_bridge_step = episode_step
        if controller.enemy_contacted and first_contact_step is None:
            first_contact_step = episode_step
        if controller.first_damage_observed and first_damage_step is None:
            first_damage_step = episode_step

    elapsed_seconds = time.perf_counter() - started_at
    final_friend_health = sum(unit.health for unit in snapshot.friends)
    final_enemy_health = sum(unit.health for unit in snapshot.enemies)
    return {
        "variant": variant.name,
        "episode": episode_index,
        "sc2_seed": episode_seed,
        **layout,
        "outcome": outcome,
        "steps": episode_step,
        "game_loop": snapshot.game_loop,
        "wall_seconds": elapsed_seconds,
        "sc2_reward": total_sc2_reward,
        "bridge_reached": controller.bridge_reached,
        "route_complete": controller.route_complete,
        "route_waypoint_index": controller.waypoint_index,
        "current_waypoint": list(controller.current_waypoint) if controller.current_waypoint else None,
        "bridge_step": first_bridge_step,
        "enemy_contact_step": first_contact_step,
        "first_attack_step": first_attack_step,
        "first_damage_step": first_damage_step,
        "initial_friends": initial_friends,
        "surviving_friends": len(snapshot.friends),
        "friendly_losses": initial_friends - len(snapshot.friends),
        "initial_enemies": initial_enemies,
        "surviving_enemies": len(snapshot.enemies),
        "enemy_kills": initial_enemies - len(snapshot.enemies),
        "initial_friend_health": initial_friend_health,
        "final_friend_health": final_friend_health,
        "initial_enemy_health": initial_enemy_health,
        "final_enemy_health": final_enemy_health,
        "final_friend_centroid": unit_centroid(snapshot.friends),
        "final_enemy_centroid": unit_centroid(snapshot.enemies),
        "move_actions": action_counts["move"],
        "attack_actions": action_counts["attack"],
        "noop_actions": action_counts["noop"],
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_variant(variant: MapVariant, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["outcome"]) for row in rows)
    total = len(rows)
    combat_wins = counts["combat_win"]
    nav_wins = counts["nav_win"]
    bridge_successes = sum(bool(row["bridge_reached"]) for row in rows)
    combat_ci = wilson_interval(combat_wins, total)
    nav_ci = wilson_interval(nav_wins, total)
    return {
        "variant": variant.name,
        "enemy_count": variant.enemy_count,
        "episodes": total,
        "outcomes": dict(sorted(counts.items())),
        "combat_win_rate": combat_wins / total if total else 0.0,
        "combat_win_rate_ci95": list(combat_ci),
        "navigation_win_rate": nav_wins / total if total else 0.0,
        "navigation_win_rate_ci95": list(nav_ci),
        "any_win_rate": (combat_wins + nav_wins) / total if total else 0.0,
        "bridge_reached_rate": bridge_successes / total if total else 0.0,
        "mean_steps": statistics.fmean(float(row["steps"]) for row in rows) if rows else 0.0,
        "mean_enemy_kills": statistics.fmean(float(row["enemy_kills"]) for row in rows) if rows else 0.0,
        "mean_surviving_friends": statistics.fmean(
            float(row["surviving_friends"]) for row in rows
        ) if rows else 0.0,
    }


def evaluate_variant(
    variant: MapVariant,
    *,
    map_dir: Path,
    episodes: int,
    seed: int,
    agent_config: AgentConfig,
    step_mul: int,
    max_steps: int,
    bot_difficulty: str,
    visualize: bool,
    realtime: bool,
    replay_dir: Path | None,
    verbose: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, raw_functions, map_size = make_environment(
        variant,
        map_dir,
        seed=seed,
        step_mul=step_mul,
        max_steps=max_steps,
        bot_difficulty=bot_difficulty,
        visualize=visualize,
        realtime=realtime,
        replay_dir=replay_dir,
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(1, episodes + 1):
            row = evaluate_episode(
                env,
                raw_functions,
                map_size,
                variant=variant,
                episode_index=episode_index,
                episode_seed=seed + episode_index - 1,
                agent_config=agent_config,
                max_steps=max_steps,
            )
            rows.append(row)
            if verbose:
                print(
                    f"[{variant.name}] episode {episode_index}/{episodes}: "
                    f"{row['outcome']} | steps={row['steps']} | "
                    f"bridge={row['bridge_reached']} | route={row['route_complete']} | "
                    f"waypoint={row['current_waypoint']} | pos={row['final_friend_centroid']} | "
                    f"kills={row['enemy_kills']} | "
                    f"survivors={row['surviving_friends']}",
                    flush=True,
                )
    finally:
        env.close()
    return rows, summarize_variant(variant, rows)


EPISODE_CSV_FIELDS = (
    "variant",
    "episode",
    "sc2_seed",
    "friendly_region",
    "enemy_region",
    "beacon_region",
    "outcome",
    "steps",
    "game_loop",
    "wall_seconds",
    "sc2_reward",
    "bridge_reached",
    "route_complete",
    "route_waypoint_index",
    "current_waypoint",
    "bridge_step",
    "enemy_contact_step",
    "first_attack_step",
    "first_damage_step",
    "initial_friends",
    "surviving_friends",
    "friendly_losses",
    "initial_enemies",
    "surviving_enemies",
    "enemy_kills",
    "initial_friend_health",
    "final_friend_health",
    "initial_enemy_health",
    "final_enemy_health",
    "final_friend_centroid",
    "final_enemy_centroid",
    "move_actions",
    "attack_actions",
    "noop_actions",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    field_list = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    output_dir: Path,
    payload: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scripted_results.json"
    episode_csv_path = output_dir / "scripted_episodes.csv"
    summary_csv_path = output_dir / "scripted_summary.csv"

    output_paths = {
        "json": str(json_path),
        "episodes_csv": str(episode_csv_path),
        "summary_csv": str(summary_csv_path),
    }
    payload["output_paths"] = output_paths

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")

    _write_csv(episode_csv_path, episode_rows, EPISODE_CSV_FIELDS)
    summary_rows = []
    for summary in summaries:
        flat = dict(summary)
        flat["outcomes"] = json.dumps(summary["outcomes"], sort_keys=True)
        flat["combat_win_rate_ci95"] = json.dumps(summary["combat_win_rate_ci95"])
        flat["navigation_win_rate_ci95"] = json.dumps(summary["navigation_win_rate_ci95"])
        summary_rows.append(flat)
    summary_fields = tuple(summary_rows[0]) if summary_rows else ("variant",)
    _write_csv(summary_csv_path, summary_rows, summary_fields)
    return output_paths


def build_parser(default_map_name: str = "V1_Base") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the state-oracle, action-matched scripted Marine baseline "
            "on Two-Bridge maps."
        )
    )
    parser.add_argument(
        "--map-name",
        default=default_map_name,
        choices=("all", *CANONICAL_VARIANT_NAMES),
        help=f"Variant to evaluate. Default: {default_map_name}",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tactic",
        choices=(
            "focus_fire",
            "focus_fire_kite",
            "per_unit_focus_fire_kite",
        ),
        default="focus_fire_kite",
    )
    parser.add_argument("--step-mul", type=int, default=DEFAULT_STEP_MUL)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--bot-difficulty", default="easy")
    parser.add_argument("--map-dir", help="Directory containing the camera-free .SC2Map files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", help="Optional output subdirectory name.")
    parser.add_argument("--no-save-results", action="store_true")
    parser.add_argument("--save-replays", action="store_true")
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(default_map_name: str = "V1_Base", argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser(default_map_name).parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.step_mul <= 0 or args.max_steps <= 0:
        raise ValueError("--step-mul and --max-steps must be positive.")

    variants = (
        [MAP_VARIANTS[name] for name in CANONICAL_VARIANT_NAMES]
        if args.map_name == "all"
        else [get_variant(args.map_name)]
    )
    map_dir = resolve_map_dir(args.map_dir)
    validate_map_files(map_dir, variants)

    started_utc = datetime.now(timezone.utc)
    run_name = args.run_name or started_utc.strftime("%Y%m%d_%H%M%S_UTC")
    run_output_dir = args.output_dir.expanduser().resolve() / run_name
    replay_root = None
    if args.save_replays:
        replay_root = (
            args.replay_dir.expanduser().resolve()
            if args.replay_dir
            else run_output_dir / "replays"
        )

    config = AgentConfig(tactic=args.tactic)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    overall_started = time.perf_counter()
    for variant_index, variant in enumerate(variants):
        variant_replay_dir = replay_root / variant.name if replay_root else None
        rows, summary = evaluate_variant(
            variant,
            map_dir=map_dir,
            episodes=args.episodes,
            seed=args.seed + variant_index * 100_000,
            agent_config=config,
            step_mul=args.step_mul,
            max_steps=args.max_steps,
            bot_difficulty=args.bot_difficulty,
            visualize=args.visualize,
            realtime=args.realtime,
            replay_dir=variant_replay_dir,
            verbose=not args.quiet,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    finished_utc = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "baseline": {
            "name": "lower_bridge_focus_fire",
            "kind": (
                "privileged_state_oracle_joint_action_matched"
                if args.tactic != "per_unit_focus_fire_kite"
                else "privileged_state_oracle_per_unit_action_matched"
            ),
            "uses_privileged_raw_state": True,
            "action_semantics": (
                "One joint command per step using noop, eight 2-unit compass moves, "
                "or in-range targeted attack with a unit-selection mask. Route commands "
                "select all living Marines."
                if args.tactic != "per_unit_focus_fire_kite"
                else
                "Simultaneous per-unit noop, eight 2-unit compass moves, or in-range "
                "targeted attacks; route movement is grouped for choke-safe formation pathing."
            ),
            "route": "visual lower bridge via cropped RAW mouths (16,17), (28,17), (36,17)",
            "tactic": args.tactic,
        },
        "configuration": {
            "variants": [variant.name for variant in variants],
            "episodes_per_variant": args.episodes,
            "seed": args.seed,
            "seed_schedule": "base + variant_index*100000 + episode_index-1",
            "step_mul": args.step_mul,
            "max_steps": args.max_steps,
            "bot_difficulty": args.bot_difficulty,
            "map_dir": str(map_dir),
            "beacon_radius": BEACON_RADIUS,
            "save_replays": bool(args.save_replays),
        },
        "started_utc": started_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "wall_seconds": time.perf_counter() - overall_started,
        "summaries": summaries,
        "episodes": all_rows,
    }

    output_paths: dict[str, str] = {}
    if not args.no_save_results:
        output_paths = save_results(run_output_dir, payload, all_rows, summaries)
        payload["output_paths"] = output_paths

    print("\nScripted baseline summary")
    for summary in summaries:
        print(
            f"  {summary['variant']}: combat={summary['combat_win_rate']:.1%}, "
            f"navigation={summary['navigation_win_rate']:.1%}, "
            f"bridge={summary['bridge_reached_rate']:.1%}, "
            f"outcomes={summary['outcomes']}"
        )
    if output_paths:
        print(f"Results: {output_paths['json']}")
    return payload


def main_for_script(script_file: str, argv: list[str] | None = None) -> dict[str, Any]:
    variant_name = Path(script_file).resolve().parent.name
    get_variant(variant_name)
    return main(default_map_name=variant_name, argv=argv)


if __name__ == "__main__":
    main()
