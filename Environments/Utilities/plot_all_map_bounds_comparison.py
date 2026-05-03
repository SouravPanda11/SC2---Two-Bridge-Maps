from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Environments.Utilities.plot_smac_map_bounds import (
    DEFAULT_SMAC_DIR,
    DEFAULT_SMACV2_DIR,
    describe_game_info,
    register_map,
)
from Environments.Utilities.plot_v2_base_agent_minimap_observation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESOLUTION,
    DEFAULT_STEP_MUL,
    import_pysc2_modules,
    install_runtime_compat,
    make_jsonable,
)


DEFAULT_MINIGAME_DIR = Path(r"C:/Program Files (x86)/StarCraft II/Maps/mini_games")
DEFAULT_FULL_MAP_PATH = Path(r"C:/Program Files (x86)/StarCraft II/Maps/Melee/Simple128.SC2Map")
DEFAULT_INSTALLED_TWOBRIDGE_PATH = Path(
    r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free/TwoBridgeMap_V2_Base.SC2Map"
)
DEFAULT_REPO_TWOBRIDGE_PATH = REPO_ROOT / "Maps" / "Camera Free" / "TwoBridgeMap_V2_Base.SC2Map"
DEFAULT_SMAC_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "smac_map_bounds" / "smac_map_bounds_summary.json"
DEFAULT_SMAC_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "smac_map_bounds" / "smac_map_bounds_summary.csv"
DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_DIR / "all_map_bounds_comparison"

GROUP_LABELS = {
    "minigame": "Mini-games",
    "smac": "SMAC",
    "smacv2": "SMAC v2",
    "twobridge": "TwoBridge",
    "full-game": "Full game",
}

GROUP_COLORS = {
    "minigame": "#1f77b4",
    "smac": "#ff7f0e",
    "smacv2": "#ff7f0e",
    "twobridge": "#2ca02c",
    "full-game": "#d62728",
}

PLOT_GROUP_LABELS = {
    "minigame": "Mini-games",
    "smac_family": "SMAC / SMAC v2",
    "twobridge": "TwoBridge",
    "full-game": "Full game",
}

PLOT_GROUP_COLORS = {
    "minigame": GROUP_COLORS["minigame"],
    "smac_family": GROUP_COLORS["smac"],
    "twobridge": GROUP_COLORS["twobridge"],
    "full-game": GROUP_COLORS["full-game"],
}


@dataclass(frozen=True)
class LiveMapSpec:
    group: str
    path: Path
    players: int
    registry_prefix: str

    @property
    def registry_name(self) -> str:
        return f"{self.registry_prefix}_{self.path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare StarCraft II mini-game, SMAC, SMAC v2, TwoBridge, and full-game "
            "map bounds. SMAC/SMAC v2 are read from the existing cached summary by "
            "default; mini-games, TwoBridge, and Simple128 are loaded through PySC2."
        )
    )
    parser.add_argument(
        "--minigame-dir",
        type=Path,
        default=DEFAULT_MINIGAME_DIR,
        help=f"Directory containing mini-game .SC2Map files. Default: {DEFAULT_MINIGAME_DIR}",
    )
    parser.add_argument(
        "--minigame-glob",
        default="*.SC2Map",
        help="Glob used inside --minigame-dir. Default: *.SC2Map",
    )
    parser.add_argument(
        "--twobridge-path",
        type=Path,
        default=DEFAULT_INSTALLED_TWOBRIDGE_PATH
        if DEFAULT_INSTALLED_TWOBRIDGE_PATH.exists()
        else DEFAULT_REPO_TWOBRIDGE_PATH,
        help="One TwoBridge .SC2Map to include. Defaults to V2_Base.",
    )
    parser.add_argument(
        "--full-map-path",
        type=Path,
        default=DEFAULT_FULL_MAP_PATH,
        help=f"Full-game melee map path. Default: {DEFAULT_FULL_MAP_PATH}",
    )
    parser.add_argument(
        "--smac-summary-json",
        type=Path,
        default=DEFAULT_SMAC_SUMMARY_JSON,
        help=f"Cached SMAC/SMAC v2 summary JSON. Default: {DEFAULT_SMAC_SUMMARY_JSON}",
    )
    parser.add_argument(
        "--smac-summary-csv",
        type=Path,
        default=DEFAULT_SMAC_SUMMARY_CSV,
        help=f"Cached SMAC/SMAC v2 summary CSV fallback. Default: {DEFAULT_SMAC_SUMMARY_CSV}",
    )
    parser.add_argument(
        "--skip-smac-cache",
        action="store_true",
        help="Do not include cached SMAC/SMAC v2 records.",
    )
    parser.add_argument(
        "--refresh-live",
        action="store_true",
        help=(
            "Reload live mini-game, TwoBridge, and full-game map bounds through PySC2. "
            "By default, the script reuses the saved combined summary when it exists."
        ),
    )
    parser.add_argument(
        "--include-smac-paths",
        action="store_true",
        help=(
            "Live-load SMAC/SMAC v2 .SC2Map files from --smac-dir/--smacv2-dir instead "
            "of relying only on the cached summary."
        ),
    )
    parser.add_argument(
        "--smac-dir",
        type=Path,
        default=DEFAULT_SMAC_DIR,
        help=f"Directory containing SMAC maps for live loading. Default: {DEFAULT_SMAC_DIR}",
    )
    parser.add_argument(
        "--smacv2-dir",
        type=Path,
        default=DEFAULT_SMACV2_DIR,
        help=f"Directory containing SMAC v2 maps for live loading. Default: {DEFAULT_SMACV2_DIR}",
    )
    parser.add_argument(
        "--smac-glob",
        default="*.SC2Map",
        help="Glob used for live SMAC/SMAC v2 loading. Default: *.SC2Map",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_SUBDIR,
        help=f"Directory for generated summary files and plots. Default: {DEFAULT_OUTPUT_SUBDIR}",
    )
    parser.add_argument(
        "--prefix",
        default="all_map_bounds_comparison",
        help="Output filename prefix. Default: all_map_bounds_comparison",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Plot format to write. Default: png",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster figure DPI. Default: 300")
    parser.add_argument(
        "--line-width",
        type=float,
        default=3.2,
        help="Playable-area rectangle line width in points. Default: 3.2",
    )
    parser.add_argument(
        "--label-font-size",
        type=float,
        default=9.0,
        help="Font size for numeric labels written on plotted extents. Default: 9.0",
    )
    parser.add_argument(
        "--title-font-size",
        type=float,
        default=18.0,
        help="Figure title font size. Default: 18.0",
    )
    parser.add_argument(
        "--axis-label-font-size",
        type=float,
        default=14.0,
        help="X/Y axis label font size. Default: 14.0",
    )
    parser.add_argument(
        "--tick-font-size",
        type=float,
        default=12.0,
        help="X/Y tick label font size. Default: 12.0",
    )
    parser.add_argument(
        "--table-font-size",
        type=float,
        default=9.5,
        help="Summary table body font size. Default: 9.5",
    )
    parser.add_argument(
        "--table-header-font-size",
        type=float,
        default=10.0,
        help="Summary table header font size. Default: 10.0",
    )
    parser.add_argument("--screen-res", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--minimap-res", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--raw-resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--step-mul", type=int, default=DEFAULT_STEP_MUL)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on the first live map load failure.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def bounds_from_csv_row(row: dict[str, str]) -> dict[str, object]:
    map_size = {
        "x": float(row["map_size_x"]),
        "y": float(row["map_size_y"]),
    }
    playable = {
        "x0": float(row["playable_x0"]),
        "y0": float(row["playable_y0"]),
        "x1": float(row["playable_x1"]),
        "y1": float(row["playable_y1"]),
        "width": float(row["playable_width"]),
        "height": float(row["playable_height"]),
    }
    return {
        "has_start_raw": True,
        "map_size": map_size,
        "playable_area": playable,
        "playable_matches_full_map": bool(
            playable["x0"] == 0
            and playable["y0"] == 0
            and playable["x1"] == map_size["x"]
            and playable["y1"] == map_size["y"]
        ),
    }


def record_from_bounds(
    group: str,
    map_name: str,
    map_file: str,
    map_path: str,
    bounds: dict[str, object],
    source: str,
) -> dict[str, object]:
    map_size = bounds.get("map_size", {})
    playable = bounds.get("playable_area", {})
    width = float(map_size.get("x", 0.0) or 0.0)
    height = float(map_size.get("y", 0.0) or 0.0)
    playable_width = float(playable.get("width", 0.0) or 0.0)
    playable_height = float(playable.get("height", 0.0) or 0.0)
    return {
        "group": group,
        "group_label": GROUP_LABELS.get(group, group),
        "map_name": map_name,
        "map_file": map_file,
        "map_path": map_path,
        "source": source,
        "status": "ok",
        "error": "",
        "bounds": bounds,
        "map_width": width,
        "map_height": height,
        "map_area": width * height,
        "playable_x0": playable.get("x0"),
        "playable_y0": playable.get("y0"),
        "playable_x1": playable.get("x1"),
        "playable_y1": playable.get("y1"),
        "playable_width": playable_width,
        "playable_height": playable_height,
        "playable_area": playable_width * playable_height,
        "playable_matches_full_map": bounds.get("playable_matches_full_map"),
    }


def error_record(spec: LiveMapSpec, exc: Exception) -> dict[str, object]:
    return {
        "group": spec.group,
        "group_label": GROUP_LABELS.get(spec.group, spec.group),
        "map_name": spec.path.stem,
        "map_file": spec.path.name,
        "map_path": str(spec.path),
        "source": "live",
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "bounds": {},
        "map_width": None,
        "map_height": None,
        "map_area": None,
        "playable_x0": None,
        "playable_y0": None,
        "playable_x1": None,
        "playable_y1": None,
        "playable_width": None,
        "playable_height": None,
        "playable_area": None,
        "playable_matches_full_map": None,
    }


def load_cached_smac_json(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in payload.get("results", []):
        group = item.get("group")
        if group not in {"smac", "smacv2"}:
            continue
        bounds = item.get("bounds", {})
        if not bounds.get("has_start_raw"):
            continue
        records.append(
            record_from_bounds(
                group=group,
                map_name=str(item.get("map_name", "")),
                map_file=str(item.get("map_file", "")),
                map_path=str(item.get("map_path", "")),
                bounds=bounds,
                source=str(path),
            )
        )
    return records


def load_cached_smac_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            group = row.get("group")
            if group not in {"smac", "smacv2"} or row.get("status") != "ok":
                continue
            records.append(
                record_from_bounds(
                    group=group,
                    map_name=str(row.get("map_name", "")),
                    map_file=str(row.get("map_file", "")),
                    map_path=str(row.get("map_path", "")),
                    bounds=bounds_from_csv_row(row),
                    source=str(path),
                )
            )
    return records


def load_cached_smac_records(args: argparse.Namespace) -> list[dict[str, object]]:
    records = load_cached_smac_json(resolve_path(args.smac_summary_json))
    if records:
        return records
    return load_cached_smac_csv(resolve_path(args.smac_summary_csv))


def load_combined_summary_records(args: argparse.Namespace) -> list[dict[str, object]]:
    summary_path = resolve_path(args.output_dir) / f"{sanitize_filename(args.prefix)}_summary.json"
    if not summary_path.exists():
        return []

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return records


def collect_live_specs(args: argparse.Namespace) -> list[LiveMapSpec]:
    specs: list[LiveMapSpec] = []

    minigame_dir = resolve_path(args.minigame_dir)
    for path in sorted(minigame_dir.glob(args.minigame_glob), key=lambda item: item.name.lower()):
        if path.is_file():
            specs.append(LiveMapSpec("minigame", path, 1, "minigame"))

    twobridge_path = resolve_path(args.twobridge_path)
    if twobridge_path.is_file():
        specs.append(LiveMapSpec("twobridge", twobridge_path, 2, "twobridge"))

    full_map_path = resolve_path(args.full_map_path)
    if full_map_path.is_file():
        specs.append(LiveMapSpec("full-game", full_map_path, 2, "fullgame"))

    if args.include_smac_paths:
        for group, directory in (("smac", args.smac_dir), ("smacv2", args.smacv2_dir)):
            directory = resolve_path(directory)
            for path in sorted(directory.glob(args.smac_glob), key=lambda item: item.name.lower()):
                if path.is_file():
                    specs.append(LiveMapSpec(group, path, 2, group))

    return specs


def create_env_for_bounds(map_name: str, players: int, sc2_env, actions, features_module, args):
    env_players = [sc2_env.Agent(sc2_env.Race.terran)]
    for _ in range(max(0, int(players) - 1)):
        env_players.append(sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy))

    return sc2_env.SC2Env(
        map_name=map_name,
        players=env_players,
        step_mul=args.step_mul,
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=args.raw_resolution,
            raw_crop_to_playable_area=True,
            feature_dimensions=features_module.Dimensions(
                screen=args.screen_res,
                minimap=args.minimap_res,
            ),
        ),
        visualize=args.visualize,
        realtime=args.realtime,
    )


def live_load_bounds(spec: LiveMapSpec, sc2_env, actions, features_module, lib_module, args):
    register_map(lib_module, spec.registry_name, str(spec.path.parent), spec.path.name, spec.players)
    env = None
    try:
        env = create_env_for_bounds(spec.registry_name, spec.players, sc2_env, actions, features_module, args)
        controller = env._controllers[0]
        before_reset = describe_game_info(controller.game_info())
        env.reset()
        after_reset = describe_game_info(controller.game_info())
        return after_reset if after_reset.get("has_start_raw") else before_reset
    finally:
        if env is not None:
            env.close()


def load_live_records(args: argparse.Namespace) -> list[dict[str, object]]:
    specs = collect_live_specs(args)
    if not specs:
        return []

    install_runtime_compat()
    sc2_env, actions, features_module, lib_module = import_pysc2_modules()

    records = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] Loading {spec.group}/{spec.path.name} ...")
        try:
            bounds = live_load_bounds(spec, sc2_env, actions, features_module, lib_module, args)
            record = record_from_bounds(
                group=spec.group,
                map_name=spec.path.stem,
                map_file=spec.path.name,
                map_path=str(spec.path),
                bounds=bounds,
                source="live",
            )
            records.append(record)
            print(
                "  "
                f"map_size=({record['map_width']:.0f}, {record['map_height']:.0f}), "
                f"playable=({record['playable_width']:.0f}, {record['playable_height']:.0f})"
            )
        except Exception as exc:
            record = error_record(spec, exc)
            records.append(record)
            print(f"  failed: {record['error']}", file=sys.stderr)
            if args.stop_on_error:
                raise

    return records


def csv_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "group": record.get("group"),
        "map_name": record.get("map_name"),
        "map_file": record.get("map_file"),
        "status": record.get("status"),
        "error": record.get("error"),
        "map_width": record.get("map_width"),
        "map_height": record.get("map_height"),
        "map_area": record.get("map_area"),
        "playable_x0": record.get("playable_x0"),
        "playable_y0": record.get("playable_y0"),
        "playable_x1": record.get("playable_x1"),
        "playable_y1": record.get("playable_y1"),
        "playable_width": record.get("playable_width"),
        "playable_height": record.get("playable_height"),
        "playable_area": record.get("playable_area"),
        "playable_matches_full_map": record.get("playable_matches_full_map"),
        "source": record.get("source"),
        "map_path": record.get("map_path"),
    }


def distinct_size_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object, object], list[str]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (
            record["group"],
            record["map_width"],
            record["map_height"],
            record["playable_width"],
            record["playable_height"],
        )
        grouped.setdefault(key, []).append(str(record["map_name"]))

    rows = []
    for (group, map_w, map_h, playable_w, playable_h), names in sorted(grouped.items()):
        rows.append(
            {
                "group": group,
                "count": len(names),
                "map_width": map_w,
                "map_height": map_h,
                "map_area": float(map_w) * float(map_h),
                "playable_width": playable_w,
                "playable_height": playable_h,
                "playable_area": float(playable_w) * float(playable_h),
                "maps": ";".join(sorted(names)),
            }
        )
    return rows


def plot_group(group: object) -> str:
    group_name = str(group)
    if group_name in {"smac", "smacv2"}:
        return "smac_family"
    return group_name


def plot_extent_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue

        group = plot_group(record["group"])
        key = (
            group,
            record["playable_x0"],
            record["playable_y0"],
            record["playable_x1"],
            record["playable_y1"],
            record["playable_width"],
            record["playable_height"],
        )
        row = grouped.setdefault(
            key,
            {
                "plot_group": group,
                "label": PLOT_GROUP_LABELS.get(group, str(record["group"])),
                "color": PLOT_GROUP_COLORS.get(group, "#555555"),
                "count": 0,
                "source_groups": set(),
                "playable_x0": float(record["playable_x0"]),
                "playable_y0": float(record["playable_y0"]),
                "playable_x1": float(record["playable_x1"]),
                "playable_y1": float(record["playable_y1"]),
                "playable_width": float(record["playable_width"]),
                "playable_height": float(record["playable_height"]),
                "playable_area": float(record["playable_area"]),
            },
        )
        row["count"] = int(row["count"]) + 1
        row["source_groups"].add(str(record["group"]))

    rows = list(grouped.values())
    for row in rows:
        row["source_groups"] = sorted(row["source_groups"])
    order = {"full-game": 0, "twobridge": 1, "minigame": 2, "smac_family": 3}
    return sorted(
        rows,
        key=lambda item: (
            order.get(str(item["plot_group"]), 9),
            -float(item["playable_area"]),
            str(item["label"]),
        ),
    )


def size_numberline_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in distinct_size_rows(records):
        group = plot_group(row["group"])
        rows.append(
            {
                **row,
                "plot_group": group,
                "label": GROUP_LABELS.get(str(row["group"]), str(row["group"])),
                "color": PLOT_GROUP_COLORS.get(group, "#555555"),
            }
        )
    return sorted(rows, key=lambda item: (float(item["map_area"]), str(item["label"])))


def summary_table_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = distinct_size_rows(records)
    by_plot_group: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_plot_group.setdefault(plot_group(row["group"]), []).append(row)

    table_rows = []
    order = [
        ("full-game", "Full game"),
        ("twobridge", "TwoBridge"),
        ("smac_family", "SMAC series"),
        ("minigame", "Mini-games"),
    ]
    for group, short_label in order:
        group_rows = sorted(by_plot_group.get(group, []), key=lambda item: float(item["map_area"]))
        if not group_rows:
            continue
        unique_sizes = []
        seen_sizes = set()
        for row in group_rows:
            key = (float(row["map_width"]), float(row["map_height"]), float(row["map_area"]))
            if key in seen_sizes:
                continue
            seen_sizes.add(key)
            unique_sizes.append(row)
        dimensions = ", ".join(
            f"{row['map_width']:.0f}x{row['map_height']:.0f}" for row in unique_sizes
        )
        if group == "minigame" and len(unique_sizes) > 1:
            min_area = min(float(row["map_area"]) for row in unique_sizes)
            max_area = max(float(row["map_area"]) for row in unique_sizes)
            areas = f"{min_area:,.0f}-{max_area:,.0f}"
        else:
            areas = ", ".join(f"{float(row['map_area']):,.0f}" for row in unique_sizes)
        count = sum(int(row["count"]) for row in group_rows)
        table_rows.append(
            {
                "group": group,
                "label": short_label,
                "dimensions": dimensions,
                "area": areas,
                "count": count,
                "color": PLOT_GROUP_COLORS.get(group, "#555555"),
            }
        )
    return table_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(records: list[dict[str, object]], args: argparse.Namespace) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok_records = [record for record in records if record.get("status") == "ok"]
    payload = {
        "record_count": len(records),
        "ok_count": len(ok_records),
        "error_count": len(records) - len(ok_records),
        "colors": GROUP_COLORS,
        "records": records,
        "distinct_sizes": distinct_size_rows(records),
    }

    json_path = args.output_dir / f"{sanitize_filename(args.prefix)}_summary.json"
    json_path.write_text(json.dumps(make_jsonable(payload), indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / f"{sanitize_filename(args.prefix)}_summary.csv"
    rows = [csv_row(record) for record in records]
    write_csv(csv_path, rows, list(csv_row({}).keys()))

    distinct_path = args.output_dir / f"{sanitize_filename(args.prefix)}_distinct_sizes.csv"
    distinct_rows = distinct_size_rows(records)
    write_csv(
        distinct_path,
        distinct_rows,
        [
            "group",
            "count",
            "map_width",
            "map_height",
            "map_area",
            "playable_width",
            "playable_height",
            "playable_area",
            "maps",
        ],
    )

    return [json_path, csv_path, distinct_path]


def add_playable_rect(axis, row: dict[str, object], args: argparse.Namespace, zorder: int) -> None:
    from matplotlib.patches import Rectangle

    color = str(row["color"])
    x0 = float(row["playable_x0"])
    y0 = float(row["playable_y0"])
    width = float(row["playable_width"])
    height = float(row["playable_height"])
    x1 = float(row["playable_x1"])
    y1 = float(row["playable_y1"])

    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=float(args.line_width),
            alpha=0.86,
            label=str(row["label"]),
            zorder=zorder,
        )
    )

    label = f"{width:.0f}x{height:.0f}"
    if int(row["count"]) > 1:
        label = f"{label} n={row['count']}"
    axis.text(
        x0 + width * 0.50,
        y1,
        label,
        ha="center",
        va="center",
        color=color,
        fontsize=float(args.label_font_size),
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.85,
            "alpha": 0.96,
        },
        clip_on=False,
        zorder=50,
    )
    axis.text(
        x1,
        y0 + height * 0.50,
        f"A={float(row['playable_area']):.0f}",
        ha="center",
        va="center",
        rotation=90,
        color=color,
        fontsize=max(6.5, float(args.label_font_size) - 1.0),
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.75,
            "alpha": 0.96,
        },
        clip_on=False,
        zorder=50,
    )


def plot_bounds_records(records: list[dict[str, object]], args: argparse.Namespace) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = plot_extent_rows(records)
    if not rows:
        return []

    max_x = max(float(row["playable_x1"]) for row in rows)
    max_y = max(float(row["playable_y1"]) for row in rows)

    fig, axis = plt.subplots(figsize=(7.4, 6.6), constrained_layout=True)
    fig.suptitle("SC2 playable map extents", fontsize=float(args.title_font_size))

    zorders = {"smac_family": 2, "minigame": 3, "twobridge": 4, "full-game": 1}
    for row in rows:
        add_playable_rect(axis, row, args, zorders.get(str(row["plot_group"]), 2))

    legend_handles = []
    seen = set()
    for row in rows:
        group = str(row["plot_group"])
        if group in seen:
            continue
        seen.add(group)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PLOT_GROUP_COLORS.get(group, "#555555"),
                linewidth=float(args.line_width),
                label=PLOT_GROUP_LABELS.get(group, group),
            )
        )
    axis.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.66, 0.56),
        fontsize=8,
        frameon=True,
    )

    axis.set_xlabel("Map width", fontsize=float(args.axis_label_font_size))
    axis.set_ylabel("Map height", fontsize=float(args.axis_label_font_size))
    axis.tick_params(axis="both", labelsize=float(args.tick_font_size))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-4, max_x + 6)
    axis.set_ylim(-4, max_y + 6)
    axis.grid(color="#d9d9d9", linewidth=0.45, alpha=0.8)

    paths = []
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    for file_format in formats:
        path = args.output_dir / f"{sanitize_filename(args.prefix)}_bounds.{file_format}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)

    plt.close(fig)
    return paths


def add_summary_table(axis, records: list[dict[str, object]], args: argparse.Namespace) -> None:
    table_rows = summary_table_rows(records)
    if not table_rows:
        return

    cell_text = [[row["label"], row["dimensions"], row["area"]] for row in table_rows]
    table = axis.table(
        cellText=cell_text,
        colLabels=["Map", "Dimensions", "Area"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.30, 0.45, 0.25],
        bbox=(0.14, 0.60, 0.72, 0.30),
        zorder=60,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(float(args.table_font_size))

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.55)
        cell.set_alpha(0.96)
        cell.set_facecolor("white")
        if row_idx == 0:
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#f2f2f2")
            cell.get_text().set_fontsize(float(args.table_header_font_size))

    for body_idx, row in enumerate(table_rows, start=1):
        color = str(row["color"])
        for col_idx in range(3):
            table[(body_idx, col_idx)].get_text().set_color(color)
            table[(body_idx, col_idx)].get_text().set_fontweight("bold")


def plot_bounds_records_with_table(records: list[dict[str, object]], args: argparse.Namespace) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = plot_extent_rows(records)
    if not rows:
        return []

    max_x = max(float(row["playable_x1"]) for row in rows)
    max_y = max(float(row["playable_y1"]) for row in rows)

    fig, axis = plt.subplots(figsize=(7.4, 6.6), constrained_layout=True)
    fig.suptitle("SC2 playable map extents", fontsize=float(args.title_font_size))

    zorders = {"smac_family": 2, "minigame": 3, "twobridge": 4, "full-game": 1}
    for row in rows:
        add_playable_rect(axis, row, args, zorders.get(str(row["plot_group"]), 2))

    add_summary_table(axis, records, args)

    axis.set_xlabel("Map width", fontsize=float(args.axis_label_font_size))
    axis.set_ylabel("Map height", fontsize=float(args.axis_label_font_size))
    axis.tick_params(axis="both", labelsize=float(args.tick_font_size))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-4, max_x + 6)
    axis.set_ylim(-4, max_y + 6)
    axis.grid(color="#d9d9d9", linewidth=0.45, alpha=0.8)

    paths = []
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    for file_format in formats:
        path = args.output_dir / f"{sanitize_filename(args.prefix)}_bounds_table.{file_format}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)

    plt.close(fig)
    return paths


def plot_size_numberline(records: list[dict[str, object]], args: argparse.Namespace) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = size_numberline_rows(records)
    if not rows:
        return []

    max_area = max(float(row["map_area"]) for row in rows)
    min_area = min(float(row["map_area"]) for row in rows)

    fig_height = max(3.4, 0.42 * len(rows) + 1.2)
    fig, axis = plt.subplots(figsize=(8.2, fig_height), constrained_layout=True)
    fig.suptitle("Distinct SC2 map areas", fontsize=13)

    axis.hlines(0, 0, max_area, color="#d6d6d6", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlim(max(min_area * 0.72, 1.0), max_area * 1.28)
    axis.set_ylim(-0.9, len(rows) - 0.25)
    axis.set_yticks([])
    axis.set_xlabel("map area (world units squared, log scale)")
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.45, alpha=0.8)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)

    for idx, row in enumerate(rows):
        area = float(row["map_area"])
        color = str(row["color"])
        y = float(idx)
        axis.vlines(area, y - 0.26, y + 0.26, color=color, linewidth=float(args.line_width) + 1.0)
        axis.scatter([area], [y], s=46, color=color, zorder=3)
        label = (
            f"{row['label']}: {row['map_width']:.0f}x{row['map_height']:.0f}, "
            f"A={area:.0f}, n={row['count']}"
        )
        axis.text(
            area,
            y + 0.30,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=float(args.label_font_size),
            fontweight="bold",
        )

    paths = []
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    for file_format in formats:
        path = args.output_dir / f"{sanitize_filename(args.prefix)}_size_numberline.{file_format}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)

    plt.close(fig)
    return paths


def plot_records(records: list[dict[str, object]], args: argparse.Namespace) -> list[Path]:
    return plot_bounds_records(records, args) + plot_bounds_records_with_table(records, args)


def main() -> int:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)

    records: list[dict[str, object]] = []
    if not args.refresh_live and not args.include_smac_paths:
        records = load_combined_summary_records(args)
        if records:
            print(f"Loaded {len(records)} records from saved combined summary.")

    if not records:
        if not args.skip_smac_cache and not args.include_smac_paths:
            smac_records = load_cached_smac_records(args)
            print(f"Loaded {len(smac_records)} cached SMAC/SMAC v2 records.")
            records.extend(smac_records)

        records.extend(load_live_records(args))

    if not records:
        print("No map records were collected.", file=sys.stderr)
        return 1

    summary_paths = write_summaries(records, args)
    plot_paths = plot_records(records, args)

    ok_count = sum(1 for record in records if record.get("status") == "ok")
    error_count = len(records) - ok_count
    print()
    print(f"Collected {ok_count} map records; {error_count} failures.")
    print("Wrote:")
    for path in summary_paths + plot_paths:
        print(f"  {path}")
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
