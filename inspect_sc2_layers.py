# ─── inspect_sc2_layers.py ────────────────────────────────────────────────
import argparse, textwrap, numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from pysc2.maps import lib
from absl import flags

# Define the custom map class
class TwoBridgeMap_Same(lib.Map):
    name = "TwoBridgeMap_Same"
    directory = "C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename = "TwoBridgeMap_Same.SC2Map"
    players = 2  # Only one controllable slot

# Fetch existing maps and force re-registration if needed
try:
    existing_maps = lib.get_maps()

    # Remove any incorrect registrations
    if "TwoBridgeMap_Same" in existing_maps:
        print("Removing previous registration of 'TwoBridgeMap_Same'.")
        del existing_maps["TwoBridgeMap_Same"]

    # Register the map correctly
    lib.get_maps()["TwoBridgeMap_Same"] = TwoBridgeMap_Same()
    print("Successfully registered 'TwoBridgeMap_Same'.")
except Exception as e:
    print(f"An error occurred while registering the map: {e}")
    exit(1)

# Verify registration
# print("Available maps:", list(lib.get_maps().keys()))

# ---------- CLI -----------------------------------------------------------
ap = argparse.ArgumentParser(description="Dump PySC2 feature‑layers for a map")
ap.add_argument("--map", "-m", required=True,
                help="Map name (must be registered with SC2 / PySC2)")
ap.add_argument("--screen", type=int, default=64,
                help="Screen / minimap resolution (default 64)")
ap.add_argument("--visualize", action="store_true",
                help="Open the game window (slower)")
args = ap.parse_args()

class TwoBridgeMap(lib.Map):
    name      = "TwoBridgeMap_Same"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_Same.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_Same", None)
lib.get_maps()["TwoBridgeMap_Same"] = TwoBridgeMap()

# ---------- make sure Abseil is happy ------------------------------------
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

# ---------- launch one‑step environment ----------------------------------
env = sc2_env.SC2Env(
    map_name=args.map,
    players=[
        sc2_env.Agent(sc2_env.Race.terran),  # Your agent (Terran)
        sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy)  # Opponent bot (Terran)
    ],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        feature_dimensions=features.Dimensions(
            screen=args.screen, minimap=args.screen
        )
    ),
    step_mul=8, game_steps_per_episode=0,
    visualize=args.visualize
)

ts   = env.reset()[0]
obs  = ts.observation

# Inspect all keys and their contents in the observation.
print("All observation keys:")
for key, value in obs.items():
    if isinstance(value, np.ndarray):
        print(f"• {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"• {key}: type={type(value)}, value={value}")

def dump(header, feat_list, arr_stack):
    print(f"\n{header}  ({arr_stack.shape[1]}×{arr_stack.shape[2]})")
    for i, (f, arr) in enumerate(zip(feat_list[:arr_stack.shape[0]], arr_stack)):
        try:
            uniq = np.unique(arr)
            small = len(uniq) <= 8
            info = f"{arr.min():3} … {arr.max():3}"
            if small:
                info = "{" + ",".join(map(str, uniq)) + "}"
            print(f" ▸ {i:2d}  {f.name:<18s}  "
                  f"dtype={arr.dtype:<8s}  {info}")
        except IndexError as e:
            print(f" ▸ {i:2d}  {f.name:<18s}  Error: {e}")

print("\n──────────────── SCREEN FEATURES ────────────────")
dump("SCREEN",  features.SCREEN_FEATURES,  obs.feature_screen)
print("Observation keys:", obs.keys())
print("Feature screen shape:", obs.feature_screen.shape)
print("Feature screen data:", obs.feature_screen)

print("\n──────────────── MINIMAP FEATURES ───────────────")
dump("MINIMAP", features.MINIMAP_FEATURES, obs.feature_minimap)

# ---------- extra misc to understand the reset state ---------------------
print("\n──────────────── MISC SCALARS ───────────────────")
print(f"game_loop      : {obs.game_loop[0]}")
print(f"minerals / gas : {obs.player.minerals} / {obs.player.vespene}")
print(f"supply         : {obs.player.food_used}/{obs.player.food_cap}")
print(f"raw_units (#)  : {len(obs.raw_units)}")

env.close()
# ──────────────────────────────────────────────────────────────────────────
