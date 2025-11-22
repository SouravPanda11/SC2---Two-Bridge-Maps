import numpy as np
import matplotlib.pyplot as plt

from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from pysc2.maps import lib

# ─────────── parse absl FLAGS once ───────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    # the string inside the list is just a dummy program name
    FLAGS(['minimap_test'])

# ─────────── map registration ───────────
class TwoBridgeMap_V2_Base(lib.Map):
    name      = "TwoBridgeMap_V2_Base"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_V2_Base.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V2_Base", None)
lib.get_maps()["TwoBridgeMap_V2_Base"] = TwoBridgeMap_V2_Base()

# ─────────── create env & grab minimap ───────────
env = sc2_env.SC2Env(
    map_name="TwoBridgeMap_V2_Base",
    players=[
        sc2_env.Agent(sc2_env.Race.terran),
        sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy),
    ],
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=64,
        feature_dimensions=features.Dimensions(screen=64, minimap=64),
    ),
    step_mul=8,
    visualize=False,
)

ts = env.reset()[0]
minimap = np.asarray(ts.observation.feature_minimap)

print("Minimap shape:", minimap.shape)
print("\nMinimap feature layers (PySC2 order):")
for idx, f in enumerate(features.MINIMAP_FEATURES):
    print(f"{idx}: {f.name}, type={f.type}, scale={f.scale}")

print("\nUnique values per channel:")
for i in range(minimap.shape[0]):
    vals = np.unique(minimap[i])
    print(f"Channel {i} ({features.MINIMAP_FEATURES[i].name}) unique values:", vals)

# OPTIONAL: visualize each layer
for i in range(minimap.shape[0]):
    plt.figure(figsize=(4, 4))
    plt.imshow(minimap[i], cmap="viridis")
    plt.title(f"Minimap layer {i}: {features.MINIMAP_FEATURES[i].name}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

env.close()
