import sys, os, glob, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch

# ───────────────────────── project path ─────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ───────────────── SB3 / Gym / Env imports ─────────────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# NAV-only env with FULL ACTION MASKS
from Environments.full_action_mask.TB_env_FAM_V2_Base_Beacon_Fixed import (
    TwoBridgeEnv, N_FRIEND
)

# ╔════════════════════════════════════════════════════════════════╗
# ║                VIS + COLOR / LEGEND SETTINGS                   ║
# ╚════════════════════════════════════════════════════════════════╝

ACTION_COLORS = {
    -1: "#ffffff",  # Dead
     0: "#1f77b4",  # No-op
     1: "#ff7f0e",  # Move
     2: "#2ca02c",  # Attack (unused in pure NAV but kept for consistency)
}
ACTION_BOUNDS = [-1.5, -0.5, 0.5, 1.5, 2.5]
ACTION_CMAP   = ListedColormap([ACTION_COLORS[k] for k in (-1, 0, 1, 2)])
ACTION_NORM   = BoundaryNorm(ACTION_BOUNDS, ACTION_CMAP.N)
ACTION_LEGEND = [
    mpatches.Patch(facecolor=ACTION_COLORS[-1], edgecolor="#888888", label="Dead"),
    mpatches.Patch(color=ACTION_COLORS[0], label="No-op"),
    mpatches.Patch(color=ACTION_COLORS[1], label="Move"),
    mpatches.Patch(color=ACTION_COLORS[2], label="Attack"),
]

def save_episode_dashboard(dest_plots_dir, agent_name, ep_idx, df, ep_r, ep_v,
                           U=None, unit_labels=None):
    """3-row dashboard: actions (stacked), reward/value, per-unit ribbons (by tag if provided)."""
    T = len(df)
    if T == 0:
        return
    t = np.arange(T)

    # Row 1 data
    noop = df["a_noop"].to_numpy()   if "a_noop"   in df else np.zeros(T, dtype=int)
    move = df["a_move"].to_numpy()   if "a_move"   in df else np.zeros(T, dtype=int)
    atck = df["a_attack"].to_numpy() if "a_attack" in df else np.zeros(T, dtype=int)

    # Row 2 data
    r = np.asarray(ep_r)[:T]
    v = np.asarray(ep_v)[:T]

    # Fallback to u0..u4 if no tag ribbon U provided
    if U is None:
        have_units = all(u in df.columns for u in ["u0", "u1", "u2", "u3", "u4"])
        if have_units:
            U = df[["u0", "u1", "u2", "u3", "u4"]].to_numpy().T
            unit_labels = [f"Unit {i}" for i in range(5)]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.0, 1.1], figure=fig)

    # Row 1: stacked actions
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(t, noop, label="No-op",  color=ACTION_COLORS[0])
    ax1.bar(t, move, bottom=noop,           label="Move",   color=ACTION_COLORS[1])
    ax1.bar(t, atck, bottom=noop+move,      label="Attack", color=ACTION_COLORS[2])
    ax1.set_ylabel("# Units")
    ax1.set_title(f"{agent_name} – Actions / Rewards / Per-Unit (Episode {ep_idx})")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(handles=ACTION_LEGEND, ncol=4, loc="upper right")

    # Row 2: reward vs value
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, r, marker='o', ls='--', label="Env Reward")
    ax2.plot(t, v, marker='x',          label="Value Estimate")
    ax2.set_ylabel("Reward / Value")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right")

    # Row 3: per-unit ribbons
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if U is not None:
        ax3.imshow(U, aspect="auto", interpolation="nearest", cmap=ACTION_CMAP, norm=ACTION_NORM)
        if unit_labels is None:
            unit_labels = [f"Unit {i}" for i in range(U.shape[0])]
        ax3.set_yticks(range(U.shape[0]))
        ax3.set_yticklabels(unit_labels)
        ax3.set_xlabel("Timestep")
        ax3.grid(False)
        ax3.set_title("Per-Unit Actions", fontsize=10)
        ax3.legend(handles=ACTION_LEGEND, ncol=4, loc="upper right", fontsize=8, frameon=True)
    else:
        ax3.axis("off")

    # Tight x-axis per episode length
    ax1.set_xlim(0, max(0, T-1))
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)

    out_path = os.path.join(dest_plots_dir, f"ep_{ep_idx}_dashboard.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

# ╔════════════════════════════════════════════════════════════════╗
# ║                       EVAL WRAPPERS                            ║
# ╚════════════════════════════════════════════════════════════════╝

class FlattenActionWrapper(Wrapper):
    """
    NAV-ONLY, FULL ACTION MASK version:
    Dict(verb, who, direction) →
        MultiDiscrete([2, 2×N_FRIEND, 9])

    Flat action layout:
      [ verb(2),
        who_0(2), who_1(2), ..., who_{N_FRIEND-1}(2),
        direction(9) ]

    We also flatten the dict action_mask from the env into a single MultiBinary.
    """
    def __init__(self, env):
        super().__init__(env)

        # MultiDiscrete layout: [verb] + [who bits] + [direction]
        self.action_space = spaces.MultiDiscrete(
            [2] + [2] * N_FRIEND + [9]
        )

        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

        self._last_mask = np.ones(flat_len, dtype=np.int8)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb":      int(a_vec[0]),
            "who":       np.asarray(a_vec[1 : 1 + N_FRIEND], np.int8),
            "direction": int(a_vec[1 + N_FRIEND]),
        }

    def _convert_mask(self, obs):
        """
        Convert env's dict masks → flat vector compatible with ActionMasker.
        Expected obs["action_mask"] structure from env:
          {
            "verb":      (2,),
            "who":       (N_FRIEND,),
            "direction": (9,),
          }
        """
        am = obs["action_mask"]  # dict of np.int8 arrays

        verb_mask      = np.asarray(am["verb"], dtype=np.int8)      # (2,)
        who_bits       = np.asarray(am["who"], dtype=np.int8)       # (N_FRIEND,)
        direction_mask = np.asarray(am["direction"], dtype=np.int8) # (9,)

        # For each who_i (∈ {0,1}), make a 2-entry mask: [allow_0, allow_1]
        who_pairs = []
        for b in who_bits:
            # 0 (don't select) always valid; 1 valid iff unit alive (b==1)
            who_pairs.extend([1, int(b)])

        flat_mask = np.concatenate([
            verb_mask,                      # 2
            np.asarray(who_pairs, np.int8), # 2*N_FRIEND
            direction_mask,                 # 9
        ], dtype=np.int8)

        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def step(self, a):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(a))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        obs = self._convert_mask(obs)
        return obs, info

    def action_masks(self):
        return self._last_mask

def mask_fn(e): return e.action_masks()

def unwrap_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e

# ╔════════════════════════════════════════════════════════════════╗
# ║                       CONFIG & PATHS                           ║
# ╚════════════════════════════════════════════════════════════════╝

AGENT_NAME = "SB_MaskPPO_FAM_Beacon_Fixed"
MAP_NAME   = "V2_Base"

MODEL_PATH = os.path.join(
    project_root, "Agents", "MaskPPO", MAP_NAME, "saved_models",
    AGENT_NAME, f"{AGENT_NAME}_final.zip"
)

EPISODES = 2
RENDER   = False

performance_root = os.path.join(
    project_root, "Agent Performance Charts", "MaskPPO", MAP_NAME, f"{AGENT_NAME}_dashboard"
)
replay_root      = os.path.join(project_root, "Replays", "MaskPPO", MAP_NAME, AGENT_NAME)
os.makedirs(performance_root, exist_ok=True)
os.makedirs(replay_root, exist_ok=True)

RESULT_KINDS = ["nav_win", "combat_win", "combat_loss", "timeout_loss", "tie"]
folders = {}  # lazy cache

def ensure_dest(kind: str) -> dict:
    """Create dirs for this result kind only when needed; memoize in `folders`."""
    if kind not in folders:
        perf_dir = os.path.join(performance_root, kind)
        d = {
            "plots" : os.path.join(perf_dir, "plots"),
            "csvs"  : os.path.join(perf_dir, "csvs"),
            "replay": os.path.join(replay_root, kind),
        }
        for p in d.values():
            os.makedirs(p, exist_ok=True)
        folders[kind] = d
    return folders[kind]

# ╔════════════════════════════════════════════════════════════════╗
# ║                         LOAD MODEL                             ║
# ╚════════════════════════════════════════════════════════════════╝

if not os.path.isfile(MODEL_PATH):
    sys.exit(f"[ERROR] Model file not found at: {MODEL_PATH}")

base_env = TwoBridgeEnv(visualize=RENDER)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskablePPO.load(MODEL_PATH, env=env, device=device)

# discount for TD reward estimate; pull from the model if available
GAMMA = float(getattr(model, "gamma", 0.99))

# ╔════════════════════════════════════════════════════════════════╗
# ║                     HELPER FUNCTIONS                           ║
# ╚════════════════════════════════════════════════════════════════╝

def _read_tag_csv(csv_path: str) -> pd.DataFrame | None:
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df

def _smooth(y: np.ndarray, win: int | None) -> np.ndarray:
    """Centered NaN-aware moving average; preserves all-NaN runs."""
    y = np.asarray(y, float)
    if win is None or win <= 1 or y.size == 0:
        return y

    k = int(win)
    if k % 2 == 0:
        k += 1  # force odd to keep centered window

    valid = ~np.isnan(y)
    y0 = np.where(valid, y, 0.0)

    kernel = np.ones(k, dtype=float)
    pad = k // 2

    s = np.convolve(np.pad(y0, (pad, pad), mode="edge"),  kernel / k, mode="valid")
    c = np.convolve(np.pad(valid.astype(float), (pad, pad), mode="edge"), kernel,      mode="valid")

    out = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    return out

def _mask_dead(data_df: pd.DataFrame, hp_df: pd.DataFrame | None) -> pd.DataFrame:
    """Set samples to NaN where HP<=0 (or HP is NaN). Column names must match."""
    if hp_df is None:
        return data_df
    out = data_df.copy()
    common = [c for c in data_df.columns if c in hp_df.columns]
    for c in common:
        hp = hp_df[c].astype(float).to_numpy()
        dead = np.isnan(hp) | (hp <= 0.0)
        y = out[c].astype(float).to_numpy()
        y[dead] = np.nan
        out[c] = y
    return out

def _timewise_nanmean(Y: np.ndarray) -> np.ndarray:
    """Mean across units per time without warnings; returns NaN where no valid samples."""
    valid = ~np.isnan(Y)
    count = valid.sum(axis=0)
    total = np.nansum(np.where(valid, Y, 0.0), axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)

def _alive_any_per_timestep(hp_df: pd.DataFrame | None) -> np.ndarray | None:
    """True where at least one friendly is alive at that timestep."""
    if hp_df is None or hp_df.empty:
        return None
    Y = np.vstack([hp_df[c].astype(float).to_numpy() for c in hp_df.columns])
    return (Y > 0.0).any(axis=0)

def _set_symmetric_ylim(ax, ys, pad_ratio: float = 0.05, min_span: float = 1e-3):
    """Make y-axis symmetric around 0 using the largest |y| across given arrays."""
    vals = []
    for y in ys:
        if y is None:
            continue
        y = np.asarray(y, dtype=float)
        if y.size:
            vals.append(np.nanmax(np.abs(y)))
    A = np.nanmax(vals) if vals else 0.0
    if not np.isfinite(A) or A < min_span:
        A = min_span
    A *= (1.0 + pad_ratio)
    ax.set_ylim(-A, A)

def _set_symmetric_ylim_clipped(
    ax,
    ys,
    pct: float = 95.0,
    pad_ratio: float = 0.05,
    min_span: float = 1e-3,
):
    """Symmetric y-limits around 0 using a percentile clip."""
    mags = []
    for y in ys:
        if y is None:
            continue
        a = np.asarray(y, dtype=float)
        if a.size:
            mags.append(np.abs(a))
    if not mags:
        return
    z = np.concatenate(mags)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return
    A = np.nanpercentile(z, pct)
    A = max(float(A), min_span)
    A *= (1.0 + pad_ratio)
    ax.set_ylim(-A, A)

def _plot_units_vs_team(
    data_df: pd.DataFrame,
    team_reduce: str = "sum",
    title: str = "",
    ylabel: str = "",
    out_png: str = "",
    smooth_win: int | None = 3,
    unit_alpha: float = 0.55,
    unit_ls: str = "--",
    unit_lw: float = 1.1,
    team_lw: float = 2.2,
    center_zero: bool = True,
    hp_gap_mask: np.ndarray | None = None,
):
    cols = [c for c in data_df.columns if data_df[c].notna().any()]
    if not cols:
        return

    Y = np.vstack([data_df[c].astype(float).to_numpy() for c in cols])

    if hp_gap_mask is not None:
        T = Y.shape[1]
        M = hp_gap_mask[:T] if hp_gap_mask.size != T else hp_gap_mask
        Y[:, ~M] = np.nan
        for c in cols:
            y = data_df[c].astype(float).to_numpy()
            y = y[:T]
            y[~M] = np.nan
            data_df[c] = y

    if team_reduce == "sum":
        team = np.nansum(Y, axis=0)
    elif team_reduce == "mean":
        team = _timewise_nanmean(Y)
    else:
        # simple median
        team = np.full(Y.shape[1], np.nan, dtype=float)
        for j in range(Y.shape[1]):
            col = Y[:, j]
            col = col[~np.isnan(col)]
            team[j] = np.median(col) if col.size else np.nan

    team_s = _smooth(team, smooth_win)

    plt.figure(figsize=(10, 3.0))
    ax = plt.gca()

    plotted_unit_series = []
    for c in cols:
        y = data_df[c].astype(float).to_numpy()
        y = _smooth(y, smooth_win)
        if np.isnan(y).all():
            continue
        label = c if len(c) <= 12 else f"{c[:4]}…{c[-4:]}"
        plt.plot(y, ls=unit_ls, lw=unit_lw, alpha=unit_alpha, label=label)
        plotted_unit_series.append(y)

    team_line, = plt.plot(team_s, lw=team_lw, label=f"Team ({team_reduce})")

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=5, fontsize="small")

    if center_zero:
        _set_symmetric_ylim(ax, plotted_unit_series + [team_s])

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _compute_health_loss_signals(friend_hp_df: pd.DataFrame, enemy_hp_df: pd.DataFrame | None):
    """
    Returns dicts of per-unit losses per step (non-negative):
      - friend_loss_per_unit[tag] : HP decreases (penalty)
      - enemy_loss_per_unit[tag]  : HP decreases (reward)
    """
    def _per_unit_loss(df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for c in df.columns:
            hp = df[c].astype(float).to_numpy()
            hp = pd.Series(hp).ffill().fillna(0.0).to_numpy()
            d = np.diff(hp, prepend=hp[0])
            loss = np.maximum(-d, 0.0)
            dead = hp <= 0.0
            loss[dead] = 0.0
            out[c] = loss
        return out

    f_losses = _per_unit_loss(friend_hp_df) if friend_hp_df is not None else {}
    e_losses = _per_unit_loss(enemy_hp_df)  if enemy_hp_df is not None else {}

    def _stack_and_team(d: dict[str, np.ndarray]):
        if not d:
            return None, np.zeros(0, dtype=float)
        M = np.vstack([v for v in d.values()])
        return M, np.nansum(M, axis=0)

    fM, f_team = _stack_and_team(f_losses)
    eM, e_team = _stack_and_team(e_losses)

    T = max((f_team.size if f_team is not None else 0),
            (e_team.size if e_team is not None else 0))
    if f_team.size == 0: f_team = np.zeros(T)
    if e_team.size == 0: e_team = np.zeros(T)
    net = e_team - f_team

    return f_losses, e_losses, f_team, e_team, net

def _plot_terminal(ax, x, y, color, label):
    ymin, ymax = ax.get_ylim()
    rng = (ymax - ymin) if np.isfinite(ymax - ymin) and (ymax > ymin) else 1.0
    y_in = (ymin <= y <= ymax)

    if y_in:
        ax.scatter(x, y, s=90, marker="*", color=color, edgecolors="none",
                   zorder=5, label=label)
    else:
        y_edge = ymax if y > ymax else ymin
        ax.scatter(x, y_edge, s=90, marker="*", color=color, edgecolors="none",
                   zorder=6, label=label)
        dy = 0.06 * rng * (1 if y > ymax else -1)
        ax.annotate(f"{y:.1f}",
                    xy=(x, y_edge),
                    xytext=(x - 8, y_edge + dy),
                    textcoords="data",
                    ha="right",
                    va="bottom" if y > ymax else "top",
                    arrowprops=dict(arrowstyle="->", lw=1, color=color),
                    color=color)

# ---------- evaluation loop -----------------------
counters = collections.Counter({k: 0 for k in RESULT_KINDS})

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False

    logs, ep_r, ep_v = [], [], []

    # per-unit (by tag) action timeline
    base = unwrap_env(env)
    cur_tags = [int(t) for t in getattr(base, "_my_tags", np.zeros(N_FRIEND, np.int64))]
    unit_hist = {t: [] for t in cur_tags if t != 0}

    # per-unit reward/HP time series (by tag)
    friend_nav_by_tag    = {}
    friend_combat_by_tag = {}   # will just be zeros for NAV env
    friend_hp_by_tag     = {}
    enemy_hp_by_tag      = {}   # empty in NAV env

    while not done:
        act, _ = model.predict(obs, deterministic=True)

        # value estimate
        obs_tensor = {k: torch.tensor(v).float().unsqueeze(0).to(model.device)
                      for k, v in obs.items()}
        with torch.no_grad():
            v_hat = model.policy.predict_values(obs_tensor).cpu().item()

        # decode current obs/act to unit labels
        vec = obs["vector"]
        fhp = vec[2 : N_FRIEND*5 : 5]
        friend_alive = (fhp > 0)

        verb     = int(act[0])                           # 0=noop, 1=move
        who_bits = np.array(act[1 : 1 + N_FRIEND], int).astype(bool)
        dir_id   = int(act[1 + N_FRIEND])               # 0 unused, 1..8

        move_ok   = (verb == 1) and who_bits.any() and (1 <= dir_id <= 8)
        attack_ok = False  # no attack in NAV env

        unit_labels = np.full(N_FRIEND, -1, np.int8)    # default DEAD
        alive_mask  = friend_alive
        unit_labels[alive_mask & ~who_bits] = 0
        if move_ok:
            unit_labels[alive_mask & who_bits] = 1
        elif attack_ok:
            unit_labels[alive_mask & who_bits] = 2
        else:
            unit_labels[alive_mask & who_bits] = 0

        # append labels by persistent TAG (not slot index)
        base = unwrap_env(env)
        cur_tags = [int(t) for t in getattr(base, "_my_tags", np.zeros(N_FRIEND, np.int64))]
        cur_tag_to_idx = {tag: i for i, tag in enumerate(cur_tags) if tag != 0}

        current_t = len(ep_r)
        for tag in cur_tags:
            if tag != 0 and tag not in unit_hist:
                unit_hist[tag] = [-1] * current_t

        for tag in list(unit_hist.keys()):
            if tag in cur_tag_to_idx:
                i = cur_tag_to_idx[tag]
                unit_hist[tag].append(int(unit_labels[i]))
            else:
                unit_hist[tag].append(-1)

        # stacked counts
        a_move   = int((unit_labels == 1).sum())
        a_attack = int((unit_labels == 2).sum())    # will be zero
        a_noop   = int((unit_labels == 0).sum())
        a_alive  = int(alive_mask.sum())

        # env step
        obs, rew, done, trunc, info = env.step(act)

        # per-unit metrics (by tag) from env
        base = unwrap_env(env)
        um = base.get_unit_metrics()  # {'friend': {tag:{nav_r, hp, nav_dist}}, 'enemy': {}}

        def _ensure_len(d, keys, length):
            for k in keys:
                if k not in d:
                    d[k] = [0.0]*length

        t_now = len(ep_r)
        f_tags = list(um.get("friend", {}).keys())
        e_tags = list(um.get("enemy", {}).keys())

        _ensure_len(friend_nav_by_tag,    f_tags, t_now)
        _ensure_len(friend_combat_by_tag, f_tags, t_now)
        _ensure_len(friend_hp_by_tag,     f_tags, t_now)
        _ensure_len(enemy_hp_by_tag,      e_tags, t_now)

        for tag in friend_nav_by_tag:
            v = um["friend"].get(tag, None)
            friend_nav_by_tag[tag].append(   float(v.get("nav_r", 0.0))    if v else 0.0)
            # combat_r doesn't exist in NAV env; keep zeros for compatibility
            friend_combat_by_tag[tag].append(float(v.get("combat_r", 0.0)) if v else 0.0)
            friend_hp_by_tag[tag].append(    float(v.get("hp", 0.0))       if v else 0.0)

        for etag in enemy_hp_by_tag:
            ve = um["enemy"].get(etag, None)
            enemy_hp_by_tag[etag].append(float(ve.get("hp", 0.0)) if ve else 0.0)

        # log step metrics (team-level from env)
        step = base.get_reward_components()
        step.update({
            "reward": rew,
            "value_estimate": v_hat,
            "a_move": a_move, "a_attack": a_attack, "a_noop": a_noop, "a_alive": a_alive,
            "A_verb": verb, "A_direction": dir_id, "A_enemy_idx": -1,   # no enemy in NAV env
            "A_selected": int((friend_alive & who_bits).sum()),
        })
        logs.append(step)
        ep_r.append(rew)
        ep_v.append(v_hat)

    # ───── finalize per-episode outputs ────────────────────────────
    res = info.get("result", "tie")
    if res not in RESULT_KINDS:
        res = "tie"
    counters[res] += 1
    dest = ensure_dest(res)

    env_r = np.asarray(ep_r, dtype=float)
    V     = np.asarray(ep_v, dtype=float)
    if len(V) >= 2:
        r_hat = V[:-1] - GAMMA * V[1:]
        r_hat = np.append(r_hat, np.nan)
        td_error = env_r[:-1] + GAMMA * V[1:] - V[:-1]
        td_error = np.append(td_error, np.nan)
    else:
        r_hat    = np.full_like(env_r, np.nan)
        td_error = np.full_like(env_r, np.nan)

    # CSV: decomposed reward components (team level)
    df = pd.DataFrame(logs)
    df["env_reward"]     = env_r
    df["value_estimate"] = V
    df["agent_r_hat"]    = r_hat
    df["td_error"]       = td_error
    df.to_csv(os.path.join(dest["csvs"], f"decomposed_ep_{ep+1}.csv"), index=False)

    # CSV: per-unit actions by tag
    tag_cols = {f"tag_{tag}": vals for tag, vals in unit_hist.items()}
    df_units_by_tag = pd.DataFrame(tag_cols)
    df_units_by_tag.to_csv(
        os.path.join(dest["csvs"], f"per_unit_actions_by_tag_ep_{ep+1}.csv"),
        index=False
    )

    # PNG: per-unit actions ribbon (by tag)
    if not df_units_by_tag.empty:
        U_tag = df_units_by_tag.to_numpy().T
        labels_tag = list(df_units_by_tag.columns)
        plt.figure(figsize=(10, 2.8))
        plt.imshow(U_tag, aspect="auto", interpolation="nearest", cmap=ACTION_CMAP, norm=ACTION_NORM)
        plt.yticks(range(U_tag.shape[0]), labels_tag)
        plt.xlabel("Timestep")
        plt.title(f"{AGENT_NAME} – Per-Unit Actions (by tag) – Episode {ep+1}")
        plt.legend(handles=ACTION_LEGEND, ncol=4, loc="upper right", fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}_actions_units_by_tag.png"))
        plt.close()
    else:
        U_tag, labels_tag = None, None

    # PNG: Episode dashboard (actions + reward/value + tag ribbon)
    save_episode_dashboard(
        dest["plots"], AGENT_NAME, ep+1, df, ep_r, ep_v,
        U=(U_tag if not df_units_by_tag.empty else None),
        unit_labels=(labels_tag if not df_units_by_tag.empty else None)
    )

    # Reward vs agent-estimated reward
    plt.figure(figsize=(10, 4))
    t = np.arange(len(env_r))
    plt.plot(t, env_r, marker='o', ls='--', label="Env Reward")
    plt.plot(t, r_hat, marker='x',           label=r"Agent-Estimated Reward ($V_t - \gamma V_{t+1}$)")
    plt.xlabel("Timestep"); plt.ylabel("Reward")
    plt.title(f"{AGENT_NAME} – Episode {ep+1} ({res}) – Reward vs Agent Estimate")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}_reward_vs_agent_est.png"))
    plt.close()

    # Reward vs Value
    plt.figure(figsize=(10, 4))
    plt.plot(env_r, label="Env Reward", marker='o', ls='--')
    plt.plot(V,     label="Value Estimate V(s)", marker='x')
    plt.xlabel("Timestep"); plt.ylabel("Reward / Value")
    plt.title(f"{AGENT_NAME} – Episode {ep+1} ({res}) – Reward vs Value")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}_reward_vs_value.png"))
    plt.close()

    # TD error
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="black", lw=1, ls="--")
    plt.plot(td_error, label="TD Error δ_t", marker='x', color="orange")
    plt.xlabel("Timestep"); plt.ylabel("TD Error")
    plt.title(f"{AGENT_NAME} – Episode {ep+1} ({res}) – TD Error")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}_td_error.png"))
    plt.close()

    # Stacked action counts
    t = np.arange(len(df))
    noop = df["a_noop"].to_numpy()   if "a_noop"   in df else np.zeros_like(t)
    move = df["a_move"].to_numpy()   if "a_move"   in df else np.zeros_like(t)
    atck = df["a_attack"].to_numpy() if "a_attack" in df else np.zeros_like(t)
    plt.figure(figsize=(10, 3.0))
    plt.bar(t, noop, label="No-op",  color=ACTION_COLORS[0])
    plt.bar(t, move, bottom=noop,          label="Move",   color=ACTION_COLORS[1])
    plt.bar(t, atck, bottom=noop+move,     label="Attack", color=ACTION_COLORS[2])
    plt.xlabel("Timestep"); plt.ylabel("# Units")
    plt.title(f"{AGENT_NAME} – Actions (Episode {ep+1})")
    plt.legend(handles=ACTION_LEGEND, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}_actions_stacked.png"))
    plt.close()

    # CSVs: per-unit reward/HP series (by tag)
    def _save_tag_timeseries(series_dict, path_prefix):
        if not series_dict:
            return
        df_ts = pd.DataFrame({f"tag_{k}": v for k, v in series_dict.items()})
        df_ts.to_csv(path_prefix + ".csv", index=False)

    csv_dir = dest["csvs"]
    _save_tag_timeseries(friend_nav_by_tag,    os.path.join(csv_dir, f"per_unit_nav_reward_ep_{ep+1}"))
    _save_tag_timeseries(friend_combat_by_tag, os.path.join(csv_dir, f"per_unit_combat_reward_ep_{ep+1}"))
    _save_tag_timeseries(friend_hp_by_tag,     os.path.join(csv_dir, f"per_unit_friend_hp_ep_{ep+1}"))
    _save_tag_timeseries(enemy_hp_by_tag,      os.path.join(csv_dir, f"per_enemy_hp_ep_{ep+1}"))

    # New plots based on those CSVs
    plots_dir = dest["plots"]

    friend_hp_csv = os.path.join(csv_dir, f"per_unit_friend_hp_ep_{ep+1}.csv")
    enemy_hp_csv  = os.path.join(csv_dir, f"per_enemy_hp_ep_{ep+1}.csv")
    nav_csv       = os.path.join(csv_dir, f"per_unit_nav_reward_ep_{ep+1}.csv")
    combat_csv    = os.path.join(csv_dir, f"per_unit_combat_reward_ep_{ep+1}.csv")

    friend_hp_df  = _read_tag_csv(friend_hp_csv)
    enemy_hp_df   = _read_tag_csv(enemy_hp_csv)
    nav_df_raw    = _read_tag_csv(nav_csv)
    combat_df_raw = _read_tag_csv(combat_csv)

    alive_mask = _alive_any_per_timestep(friend_hp_df)

    if nav_df_raw is not None:
        nav_df = _mask_dead(nav_df_raw, friend_hp_df)
        _plot_units_vs_team(
            nav_df, team_reduce="mean",
            title=f"{AGENT_NAME} – Episode {ep+1} – Navigation Distance-Δ Reward",
            ylabel="nav_r",
            out_png=os.path.join(plots_dir, f"ep_{ep+1}_nav_distance_reward_units_vs_team.png"),
            smooth_win=11,
            center_zero=True,
            hp_gap_mask=alive_mask,
        )

    if combat_df_raw is not None:
        combat_df = _mask_dead(combat_df_raw, friend_hp_df)
        _plot_units_vs_team(
            combat_df, team_reduce="mean",
            title=f"{AGENT_NAME} – Episode {ep+1} – Combat Distance-Δ Reward",
            ylabel="combat_r",
            out_png=os.path.join(plots_dir, f"ep_{ep+1}_combat_distance_reward_units_vs_team.png"),
            smooth_win=11,
            center_zero=True,
            hp_gap_mask=alive_mask,
        )

    # Overall nav vs overall combat (combat will be ~0 for NAV env)
    T = len(df); t = np.arange(T)
    nav_r    = df["nav_r"].to_numpy()    if "nav_r"    in df.columns else None
    combat_r = df["combat_r"].to_numpy() if "combat_r" in df.columns else None
    term_r   = df["term_r"].to_numpy()   if "term_r"   in df.columns else None

    nav_terminal    = np.zeros(T, dtype=float)
    combat_terminal = np.zeros(T, dtype=float)
    if term_r is not None:
        if res == "nav_win":
            nav_terminal[-1] = term_r[-1]
        elif res in ("combat_win", "combat_loss", "timeout_loss", "victory", "defeat"):
            combat_terminal[-1] = term_r[-1]

    overall_nav    = None if nav_r    is None else (nav_r    + nav_terminal)
    overall_combat = None if combat_r is None else (combat_r + combat_terminal)

    if (overall_nav is not None) or (overall_combat is not None):
        plt.figure(figsize=(10, 3.4))
        ax = plt.gca()

        nav_line_for_scale    = None if overall_nav    is None else overall_nav.copy()
        combat_line_for_scale = None if overall_combat is None else overall_combat.copy()
        if nav_line_for_scale is not None and nav_terminal[-1] != 0:
            nav_line_for_scale[-1] = np.nan
        if combat_line_for_scale is not None and combat_terminal[-1] != 0:
            combat_line_for_scale[-1] = np.nan

        if nav_line_for_scale is not None:
            plt.plot(t, _smooth(nav_line_for_scale, 3), lw=2.4,
                     label="Overall Nav (distance)")
        if combat_line_for_scale is not None:
            plt.plot(t, _smooth(combat_line_for_scale, 3), lw=2.4,
                     label="Overall Combat (distance + HP + kill/loss)")

        if overall_nav is not None and nav_terminal[-1] != 0:
            _plot_terminal(ax, t[-1], overall_nav[-1], color="C0", label="Nav terminal")

        if overall_combat is not None and combat_terminal[-1] != 0:
            _plot_terminal(ax, t[-1], overall_combat[-1], color="C1", label="Combat terminal")

        plt.xlabel("Timestep"); plt.ylabel("Reward")
        plt.title(f"{AGENT_NAME} – Episode {ep+1} – Overall Nav vs Overall Combat")
        plt.legend()
        _set_symmetric_ylim_clipped(ax, [nav_line_for_scale, combat_line_for_scale], pct=95)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ep_{ep+1}_overall_nav_vs_combat.png"))
        plt.close()

    # Friendly HP: dashed per-unit, solid team sum
    if friend_hp_df is not None:
        _plot_units_vs_team(
            friend_hp_df, team_reduce="sum",
            title=f"{AGENT_NAME} – Episode {ep+1} – Friendly HP (per-unit dashed, team solid)",
            ylabel="HP",
            out_png=os.path.join(plots_dir, f"ep_{ep+1}_friendly_hp_units_vs_team.png"),
            smooth_win=1,
        )

    # Enemy HP: empty in NAV env → plot will be skipped if enemy_hp_df is None
    if enemy_hp_df is not None:
        _plot_units_vs_team(
            enemy_hp_df, team_reduce="sum",
            title=f"{AGENT_NAME} – Episode {ep+1} – Enemy HP (per-unit dashed, team solid)",
            ylabel="HP",
            out_png=os.path.join(plots_dir, f"ep_{ep+1}_enemy_hp_units_vs_team.png"),
            smooth_win=1,
        )

    # Health-based reward view (will be trivial in NAV env, but kept for symmetry)
    if (friend_hp_df is not None) or (enemy_hp_df is not None):
        f_losses, e_losses, f_team, e_team, net = _compute_health_loss_signals(friend_hp_df, enemy_hp_df)
        T = max(e_team.size, f_team.size, net.size) if net is not None else 0
        if T > 0:
            plt.figure(figsize=(10, 3.2))
            ax = plt.gca()
            t = np.arange(T)

            for d in (e_losses,):
                for _, y in d.items():
                    y = _smooth(y, 3)
                    plt.plot(t, y, ls="--", lw=0.9, alpha=0.25, label=None)

            for d in (f_losses,):
                for _, y in d.items():
                    y = _smooth(y, 3)
                    plt.plot(t, -y, ls="--", lw=0.9, alpha=0.20, label=None)

            enemy_line,  = plt.plot(t, _smooth(e_team, 3), ls="--", lw=2.0, label="Enemy HP loss (+)")
            friend_line, = plt.plot(t, -_smooth(f_team, 3), ls="--", lw=2.0, label="Friendly HP loss (−)")
            net_line,    = plt.plot(t, _smooth(net, 3), lw=2.6, label="Net health reward")

            plt.xlabel("Timestep")
            plt.ylabel("Reward units")
            plt.title(f"{AGENT_NAME} – Episode {ep+1} – Health-Based Reward (net solid)")
            plt.legend(ncol=3)
            _set_symmetric_ylim_clipped(ax, [e_team, -f_team, net], pct=95)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"ep_{ep+1}_health_based_reward.png"))
            plt.close()

    # Replays (if enabled)
    base = unwrap_env(env)
    if hasattr(base, "_env") and hasattr(base._env, "save_replay"):
        base._env.save_replay(dest["replay"], prefix=f"ep_{ep+1}")
        newest = sorted(
            glob.glob(os.path.join(dest["replay"], f"ep_{ep+1}_*.SC2Replay")),
            key=os.path.getmtime, reverse=True
        )
        if newest:
            os.rename(newest[0], os.path.join(dest["replay"], f"ep_{ep+1}.SC2Replay"))

    print(f"[{ep+1}/{EPISODES}] result: {res}")

env.close()

# ───────── Summary bar chart ─────────
labels, values = zip(*[(k, counters[k]) for k in RESULT_KINDS])
plt.figure(figsize=(7, 4))
plt.bar(labels, values); plt.xticks(rotation=30)
plt.ylabel("# episodes out of " + str(EPISODES))
plt.title("Agent performance"); plt.tight_layout()
plt.savefig(os.path.join(performance_root, f"{AGENT_NAME}_performance_{EPISODES}_ep.png"))
plt.show()

print("\nEpisode counts:", dict(counters))
win_pct = 100 * (counters["nav_win"] + counters["combat_win"]) / EPISODES
print(f"Win rate: {win_pct:.1f}%")
