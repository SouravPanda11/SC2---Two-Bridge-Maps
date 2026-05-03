from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Environments" / "Utilities" / "outputs"
DEFAULT_MAP_NAME = "TwoBridgeMap_V2_Base"
DEFAULT_MAP_FILE = "TwoBridgeMap_V2_Base.SC2Map"
DEFAULT_MAP_DIR = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
DEFAULT_RESOLUTION = 64
DEFAULT_STEP_MUL = 8
N_FRIEND = 5


def install_runtime_compat() -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            random.shuffle([], lambda: 0.5)
        return
    except TypeError:
        pass

    def compat_shuffle(seq, rand=None):
        if rand is None:
            return random._inst.shuffle(seq)
        for idx in range(len(seq) - 1, 0, -1):
            swap_idx = int(rand() * (idx + 1))
            seq[idx], seq[swap_idx] = seq[swap_idx], seq[idx]

    random.shuffle = compat_shuffle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the exact two-channel minimap observation used by the "
            "NS MaskPPO V2_Base Two Bridge environment: pathable "
            "and player_relative."
        )
    )
    parser.add_argument(
        "--source",
        choices=("ns", "direct"),
        default="ns",
        help=(
            "Observation source. 'ns' uses Environments.NS_AM_RM_mean.V2_Base_NS, "
            "'direct' reads PySC2 feature_minimap with the same interface settings. "
            "Default: ns"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default="v2_base_agent_minimap_observation",
        help="Output filename prefix. Default: v2_base_agent_minimap_observation",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Figure format to write. Default: png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster figure DPI. Default: 300",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of no-op environment steps after reset before capture. Default: 0",
    )
    parser.add_argument(
        "--screen-res",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Feature screen resolution requested from PySC2. Default: 64",
    )
    parser.add_argument(
        "--minimap-res",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Feature minimap resolution requested from PySC2. Default: 64",
    )
    parser.add_argument(
        "--raw-resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Raw action/observation resolution requested from PySC2. Default: 64",
    )
    parser.add_argument(
        "--map-name",
        default=DEFAULT_MAP_NAME,
        help=f"Direct PySC2 map registry name. Default: {DEFAULT_MAP_NAME}",
    )
    parser.add_argument(
        "--map-file",
        default=DEFAULT_MAP_FILE,
        help=f"Direct PySC2 map filename. Default: {DEFAULT_MAP_FILE}",
    )
    parser.add_argument(
        "--map-dir",
        default=DEFAULT_MAP_DIR,
        help=f"Direct PySC2 map directory. Default: {DEFAULT_MAP_DIR}",
    )
    parser.add_argument(
        "--map-path",
        type=Path,
        help="Optional direct path to a .SC2Map file. Overrides --map-name/--map-file/--map-dir.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the SC2 visualizer while sampling the observation.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run SC2 in realtime while sampling the observation.",
    )
    parser.add_argument(
        "--origin",
        choices=("upper", "lower"),
        default="upper",
        help=(
            "imshow origin. 'upper' shows the tensor exactly as indexed by "
            "row/column; 'lower' flips y for a map-like Cartesian view. Default: upper"
        ),
    )
    parser.add_argument(
        "--title",
        default="TwoBridgeMap V2_Base agent minimap observation",
        help="Figure title. Use an empty string to omit. Default: descriptive title.",
    )
    return parser.parse_args()


def import_pysc2_modules():
    try:
        from absl import flags
        from pysc2.env import sc2_env
        from pysc2.lib import actions, features
        from pysc2.maps import lib
    except Exception as exc:
        raise RuntimeError(
            "Failed to import PySC2. Run this with the project SC2 Python "
            "environment, for example `TBMsc2\\Scripts\\python.exe`. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    parsed_flags = flags.FLAGS
    if not parsed_flags.is_parsed():
        parsed_flags([sys.argv[0]])

    return sc2_env, actions, features, lib


def minimap_feature_indices(features_module) -> tuple[int, int]:
    return (
        int(features_module.MINIMAP_FEATURES.pathable.index),
        int(features_module.MINIMAP_FEATURES.player_relative.index),
    )


def register_direct_map(lib_module, map_name: str, map_dir: str, map_file: str) -> None:
    map_cls = type(
        map_name,
        (lib_module.Map,),
        {
            "name": map_name,
            "directory": map_dir,
            "filename": map_file,
            "players": 2,
        },
    )
    lib_module.get_maps().pop(map_name, None)
    lib_module.get_maps()[map_name] = map_cls()


def resolve_direct_map(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.map_path is None:
        return args.map_name, args.map_dir, args.map_file

    map_path = args.map_path.expanduser().resolve()
    return map_path.stem, str(map_path.parent), map_path.name


def noop_steps_ns(env, steps: int):
    obs = None
    info = {}
    action = np.zeros(N_FRIEND, dtype=np.int64)
    for _ in range(steps):
        obs, _, done, _, info = env.step(action)
        if done:
            break
    return obs, info


def sample_from_ns(args: argparse.Namespace, features_module) -> tuple[np.ndarray, dict[str, object]]:
    from Environments.NS_AM_RM_mean.V2_Base_NS import TwoBridgeEnv

    env = TwoBridgeEnv(
        screen_res=args.screen_res,
        visualize=args.visualize,
        realtime=args.realtime,
    )
    try:
        obs, _ = env.reset()
        step_obs, info = noop_steps_ns(env, args.steps)
        if step_obs is not None:
            obs = step_obs
        metadata = {
            "source": "ns",
            "environment": "Environments.NS_AM_RM_mean.V2_Base_NS.TwoBridgeEnv",
            "map_name": DEFAULT_MAP_NAME,
            "steps_after_reset": int(args.steps),
            "info": make_jsonable(info),
        }
        return np.asarray(obs["minimap"], dtype=np.uint8), metadata
    finally:
        env.close()


def sample_direct(args: argparse.Namespace, sc2_env, actions, features_module, lib_module):
    map_name, map_dir, map_file = resolve_direct_map(args)
    register_direct_map(lib_module, map_name, map_dir, map_file)

    env = sc2_env.SC2Env(
        map_name=map_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy),
        ],
        step_mul=DEFAULT_STEP_MUL,
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
    try:
        ts = env.reset()[0]
        for _ in range(args.steps):
            ts = env.step([actions.RAW_FUNCTIONS.no_op()])[0]
            if ts.last():
                break

        pathable_idx, player_relative_idx = minimap_feature_indices(features_module)
        feature_minimap = np.asarray(ts.observation.feature_minimap, dtype=np.uint8)
        metadata = {
            "source": "direct",
            "environment": "pysc2.env.sc2_env.SC2Env",
            "map_name": map_name,
            "map_dir": map_dir,
            "map_file": map_file,
            "steps_after_reset": int(args.steps),
        }
        return feature_minimap[[pathable_idx, player_relative_idx]], metadata
    finally:
        env.close()


def sample_minimap(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object], object]:
    install_runtime_compat()
    sc2_env, actions, features_module, lib_module = import_pysc2_modules()

    if args.source == "ns":
        minimap, metadata = sample_from_ns(args, features_module)
    else:
        minimap, metadata = sample_direct(args, sc2_env, actions, features_module, lib_module)

    if minimap.ndim != 3 or minimap.shape[0] != 2:
        raise ValueError(f"Expected minimap shape (2, H, W), received {minimap.shape}.")

    pathable_idx, player_relative_idx = minimap_feature_indices(features_module)
    metadata.update(
        {
            "tensor_shape": tuple(int(v) for v in minimap.shape),
            "channel_0": {
                "name": "pathable",
                "pysc2_minimap_feature_index": int(pathable_idx),
            },
            "channel_1": {
                "name": "player_relative",
                "pysc2_minimap_feature_index": int(player_relative_idx),
            },
        }
    )
    return minimap, metadata, features_module


def make_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    return value


def unique_counts(array: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(array, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def player_relative_labels(features_module) -> dict[int, str]:
    labels = {0: "none", 1: "self", 2: "ally", 3: "neutral", 4: "enemy"}
    enum_cls = getattr(features_module, "PlayerRelative", None)
    if enum_cls is None:
        return labels

    try:
        enum_items = list(enum_cls)
    except TypeError:
        enum_items = []

    for item in enum_items:
        try:
            labels[int(item)] = item.name.lower()
        except Exception:
            continue
    return labels


def build_rgb_composite(pathable: np.ndarray, player_relative: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pathable.shape, 3), dtype=np.float32)

    rgb[pathable > 0] = (0.92, 0.92, 0.88)
    rgb[pathable == 0] = (0.10, 0.12, 0.14)

    colors = {
        1: (0.05, 0.36, 0.92),  # self
        2: (0.18, 0.64, 0.34),  # ally
        3: (0.90, 0.75, 0.12),  # neutral/beacon
        4: (0.86, 0.16, 0.13),  # enemy
    }
    for value, color in colors.items():
        rgb[player_relative == value] = color
    return rgb


def plot_minimap(minimap: np.ndarray, metadata: dict[str, object], features_module, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    pathable = minimap[0]
    player_relative = minimap[1]
    labels = player_relative_labels(features_module)

    path_cmap = ListedColormap(["#1b1f23", "#f0f0ea"])
    path_norm = BoundaryNorm([-0.5, 0.5, 1.5], path_cmap.N)
    pr_cmap = ListedColormap(["#161a1d", "#0d5be1", "#2ca25f", "#e0b91a", "#d33f32"])
    pr_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], pr_cmap.N)
    composite = build_rgb_composite(pathable, player_relative)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.03)

    axes[0].imshow(pathable, cmap=path_cmap, norm=path_norm, interpolation="nearest", origin=args.origin)
    axes[0].set_title("Channel 0: pathable")
    axes[0].legend(
        handles=[
            Patch(facecolor="#1b1f23", edgecolor="black", label="0 blocked"),
            Patch(facecolor="#f0f0ea", edgecolor="black", label="1 pathable"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.26),
        frameon=False,
        ncol=2,
        fontsize=8,
    )

    axes[1].imshow(
        player_relative,
        cmap=pr_cmap,
        norm=pr_norm,
        interpolation="nearest",
        origin=args.origin,
    )
    axes[1].set_title("Channel 1: player_relative")
    present_values = sorted(int(v) for v in np.unique(player_relative))
    handles = [
        Patch(
            facecolor=pr_cmap(pr_norm(value)),
            edgecolor="black",
            label=f"{value} {labels.get(value, 'unknown')}",
        )
        for value in present_values
    ]
    axes[1].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False,
        ncol=2,
        fontsize=8,
    )

    axes[2].imshow(composite, interpolation="nearest", origin=args.origin)
    axes[2].set_title("Composite")
    axes[2].legend(
        handles=[
            Patch(facecolor="#f0f0ea", edgecolor="black", label="pathable"),
            Patch(facecolor="#1b1f23", edgecolor="black", label="blocked"),
            Patch(facecolor="#0d5be1", edgecolor="black", label="self"),
            Patch(facecolor="#e0b91a", edgecolor="black", label="neutral/beacon"),
            Patch(facecolor="#d33f32", edgecolor="black", label="enemy"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.46),
        frameon=False,
        ncol=2,
        fontsize=8,
    )

    for axis in axes:
        axis.set_xlabel("minimap x pixel")
        axis.set_ylabel("minimap y pixel")
        axis.set_xticks(np.arange(0, minimap.shape[2], 8))
        axis.set_yticks(np.arange(0, minimap.shape[1], 8))
        axis.tick_params(labelsize=7)
        axis.grid(color="white", alpha=0.08, linewidth=0.3)

    subtitle = (
        f"source={metadata['source']}, tensor shape={tuple(metadata['tensor_shape'])}, "
        "agent input order=[pathable, player_relative]"
    )
    fig.text(0.5, -0.02, subtitle, ha="center", va="center", fontsize=9)
    return fig


def write_outputs(minimap: np.ndarray, metadata: dict[str, object], features_module, args) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pathable = minimap[0]
    player_relative = minimap[1]
    metadata = dict(metadata)
    metadata["pathable_unique_counts"] = unique_counts(pathable)
    metadata["player_relative_unique_counts"] = unique_counts(player_relative)
    metadata["player_relative_labels"] = {
        str(key): value for key, value in player_relative_labels(features_module).items()
    }

    written = []
    npy_path = args.output_dir / f"{args.prefix}_tensor.npy"
    np.save(npy_path, minimap)
    written.append(npy_path)

    json_path = args.output_dir / f"{args.prefix}_summary.json"
    json_path.write_text(json.dumps(make_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    written.append(json_path)

    fig = plot_minimap(minimap, metadata, features_module, args)
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    for file_format in formats:
        figure_path = args.output_dir / f"{args.prefix}.{file_format}"
        fig.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
        written.append(figure_path)

    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def main() -> int:
    args = parse_args()
    minimap, metadata, features_module = sample_minimap(args)
    written = write_outputs(minimap, metadata, features_module, args)

    print("Captured V2_Base agent minimap observation.")
    print(f"Tensor shape: {tuple(int(v) for v in minimap.shape)}")
    print(f"Pathable unique counts: {unique_counts(minimap[0])}")
    print(f"Player_relative unique counts: {unique_counts(minimap[1])}")
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
