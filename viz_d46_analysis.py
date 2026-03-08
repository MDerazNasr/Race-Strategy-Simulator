"""
D46 Policy Analysis — what the multi-agent champion actually learned.

Four panels:
  A  Weight heatmap — D46's first policy layer (obs → 128 neurons)
  B  Column importance — D39 vs D46 side-by-side (mean |weight| per obs dim)
  C  track_gap weight evolution — across D39 → D41 → D43 → D44 → D45 → D46
  D  Head-to-head position trace — speed + track_gap over a full D46 episode

Saves to: plots/d46_analysis.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from env.f1_multi_env import F1MultiAgentEnv

# ── Constants ─────────────────────────────────────────────────────────────────
device = "cpu"

OBS_LABELS = [
    "speed", "hdg_err", "lat_err", "sin(hdg)", "cos(hdg)",
    "curv_near", "curv_mid", "curv_far", "progress", "v_y", "yaw_rate",
    "track_gap", "opp_speed",
]

# track_gap column index in 13D obs
TRACK_GAP_COL = 11

EVOLUTION = [
    ("D39\n(opp=22)", "rl/ppo_multi_agent_d39.zip",  6025),
    ("D41\n(opp=27)", "rl/ppo_multi_agent_d41.zip",    69),
    ("D43\n(opp=25)", "rl/ppo_multi_agent_d43.zip",   275),
    ("D44\n(pos×4)",  "rl/ppo_multi_agent_d44.zip",  1834),
    ("D45\n(8 laps)", "rl/ppo_multi_agent_d45.zip",  3727),
    ("D46\n★ BEST",   "rl/ppo_multi_agent_d46.zip",  7943),
]

BG    = "#0D1117"
PANEL = "#161B22"
GRID  = "#1C2128"
SPINE = "#30363D"
TEXT  = "#CCCCCC"
MUTED = "#AAAAAA"

BLUE   = "#4FC3F7"
ORANGE = "#FFB74D"
PURPLE = "#CE93D8"
GREEN  = "#81C784"
RED    = "#EF5350"

# ── Helper: extract importance vector from a loaded model ─────────────────────

def col_importance(model):
    """Mean |weight| per obs column in first policy layer. Returns (obs_dim,) array."""
    w = model.policy.mlp_extractor.policy_net[0].weight  # (128, obs_dim)
    return w.abs().mean(dim=0).detach().cpu().numpy()


def track_gap_weight(model):
    """Mean |weight| for the track_gap column (col 11)."""
    w = model.policy.mlp_extractor.policy_net[0].weight  # (128, obs_dim)
    return w[:, TRACK_GAP_COL].abs().mean().item()


# ── Load models ───────────────────────────────────────────────────────────────
print("[Viz] Loading D39 and D46...")
model_d39 = PPO.load(str(project_root / "rl" / "ppo_multi_agent_d39.zip"), device=device)
model_d46 = PPO.load(str(project_root / "rl" / "ppo_multi_agent_d46.zip"), device=device)

print("[Viz] Loading evolution models...")
evo_models = []
for label, path, reward in EVOLUTION:
    m = PPO.load(str(project_root / path), device=device)
    evo_models.append((label, m, reward))
    print(f"  {label.replace(chr(10), ' ')}: track_gap weight = {track_gap_weight(m):.6f}")

# ── Collect Panel D trajectory ─────────────────────────────────────────────────
print("[Viz] Collecting D46 episode trajectory...")
env_d46 = F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)
obs, _ = env_d46.reset(options={"fixed_start": True})
steps_list, speeds, track_gaps = [], [], []
for step in range(2000):
    action, _ = model_d46.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_d46.step(action)
    steps_list.append(step)
    speeds.append(info["speed"])
    track_gaps.append(float(obs[TRACK_GAP_COL]))   # obs[11] = track_gap
    if terminated or truncated:
        break
steps_arr = np.array(steps_list)
speeds_arr = np.array(speeds)
gaps_arr   = np.array(track_gaps)
print(f"  → {len(steps_arr)} steps, avg speed {speeds_arr.mean():.2f} m/s")
print(f"  → Ego ahead {(gaps_arr < 0).mean()*100:.1f}% of steps")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle(
    "D46 Multi-Agent Champion — Policy Analysis",
    color="white", fontsize=15, fontweight="bold", y=0.99,
)

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.42,
    wspace=0.32,
    top=0.93, bottom=0.07, left=0.06, right=0.97,
)

ax_A = fig.add_subplot(gs[0, 0])  # weight heatmap
ax_B = fig.add_subplot(gs[0, 1])  # column importance
ax_C = fig.add_subplot(gs[1, 0])  # evolution
ax_D = fig.add_subplot(gs[1, 1])  # position trace (split into 2 sub-axes below)

# Remove ax_D placeholder; replace with two sub-axes
ax_D.remove()
gs_D = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[1, 1], hspace=0.08,
)
ax_D1 = fig.add_subplot(gs_D[0])  # speed
ax_D2 = fig.add_subplot(gs_D[1])  # track_gap

for ax in [ax_A, ax_B, ax_C, ax_D1, ax_D2]:
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)

# ── Panel A: Weight heatmap ────────────────────────────────────────────────────
W = model_d46.policy.mlp_extractor.policy_net[0].weight.detach().cpu().numpy()  # (128, 13)
# Subsample neurons for readability (every 4th row)
W_sub = W[::4, :]  # (32, 13)
vmax = np.abs(W).max() * 0.6   # cap at 60th percentile-ish for contrast

im = ax_A.imshow(W_sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax_A.set_xticks(range(len(OBS_LABELS)))
ax_A.set_xticklabels(OBS_LABELS, rotation=45, ha="right", fontsize=7, color=TEXT)
ax_A.set_yticks([])
ax_A.set_xlabel("Observation dimension", color=MUTED, fontsize=9)
ax_A.set_ylabel("Policy neurons (every 4th)", color=MUTED, fontsize=9)
ax_A.set_title("A  First-layer policy weights — D46", color="white", fontsize=10, pad=6)

# Colorbar
cbar = fig.colorbar(im, ax=ax_A, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors=MUTED, labelsize=7)
cbar.outline.set_edgecolor(SPINE)

# Highlight track_gap column
ax_A.axvline(TRACK_GAP_COL - 0.5, color=PURPLE, linewidth=1.5, alpha=0.7)
ax_A.axvline(TRACK_GAP_COL + 0.5, color=PURPLE, linewidth=1.5, alpha=0.7)
ax_A.text(
    TRACK_GAP_COL, -3, "track_gap",
    color=PURPLE, fontsize=7, ha="center", va="top", rotation=45,
)

# ── Panel B: Column importance D39 vs D46 ─────────────────────────────────────
imp_d39 = col_importance(model_d39)
imp_d46 = col_importance(model_d46)

x = np.arange(len(OBS_LABELS))
width = 0.35

bars_d39 = ax_B.bar(x - width/2, imp_d39, width, label="D39 (no strategy)", color=BLUE,   alpha=0.75)
bars_d46 = ax_B.bar(x + width/2, imp_d46, width, label="D46 (positional)",  color=ORANGE, alpha=0.75)

# Highlight track_gap bars
ax_B.bar([TRACK_GAP_COL - width/2], [imp_d39[TRACK_GAP_COL]], width, color=BLUE,   alpha=1.0, edgecolor="white", linewidth=0.8)
ax_B.bar([TRACK_GAP_COL + width/2], [imp_d46[TRACK_GAP_COL]], width, color=ORANGE, alpha=1.0, edgecolor="white", linewidth=0.8)

ax_B.set_xticks(x)
ax_B.set_xticklabels(OBS_LABELS, rotation=45, ha="right", fontsize=7, color=TEXT)
ax_B.set_ylabel("Mean |weight|", color=MUTED, fontsize=9)
ax_B.set_title("B  Column importance — D39 vs D46", color="white", fontsize=10, pad=6)
ax_B.legend(fontsize=8, facecolor=BG, edgecolor=SPINE, labelcolor="white", loc="upper right")

# Annotate track_gap change
tg_change = (imp_d46[TRACK_GAP_COL] - imp_d39[TRACK_GAP_COL]) / (imp_d39[TRACK_GAP_COL] + 1e-8) * 100
ax_B.annotate(
    f"track_gap\n+{tg_change:.0f}%",
    xy=(TRACK_GAP_COL + width/2, imp_d46[TRACK_GAP_COL]),
    xytext=(TRACK_GAP_COL + 1.5, imp_d46[TRACK_GAP_COL] + 0.002),
    fontsize=7.5, color=PURPLE,
    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8),
)

# ── Panel C: track_gap weight evolution ───────────────────────────────────────
evo_labels  = [e[0] for e in evo_models]
evo_weights = [track_gap_weight(e[1]) for e in evo_models]
evo_rewards = [e[2] for e in evo_models]

evo_colors = [BLUE, RED, ORANGE, GREEN, PURPLE, PURPLE]

ax_C.plot(range(len(evo_labels)), evo_weights, "-o",
          color=PURPLE, linewidth=1.8, markersize=7, zorder=3)

# Color individual markers
for i, (w, c) in enumerate(zip(evo_weights, evo_colors)):
    ax_C.plot(i, w, "o", color=c, markersize=9, zorder=4)

# Annotate with reward
for i, (w, r) in enumerate(zip(evo_weights, evo_rewards)):
    va = "bottom" if i % 2 == 0 else "top"
    dy = 0.001 if va == "bottom" else -0.001
    ax_C.annotate(
        f"reward\n{r:,}",
        xy=(i, w), xytext=(i, w + dy),
        fontsize=7, color=MUTED, ha="center", va=va,
    )

ax_C.set_xticks(range(len(evo_labels)))
ax_C.set_xticklabels(evo_labels, fontsize=8, color=TEXT)
ax_C.set_ylabel("Mean |weight| on track_gap (col 11)", color=MUTED, fontsize=9)
ax_C.set_title("C  Positional awareness — track_gap weight evolution", color="white", fontsize=10, pad=6)
ax_C.axhline(evo_weights[0], color=BLUE, linestyle=":", linewidth=0.8, alpha=0.5, label=f"D39 baseline ({evo_weights[0]:.4f})")
ax_C.legend(fontsize=7, facecolor=BG, edgecolor=SPINE, labelcolor="white")

# ── Panel D: Head-to-head position trace ──────────────────────────────────────
# Speed subplot
ax_D1.plot(steps_arr, speeds_arr, color=BLUE, linewidth=1.0, zorder=3)
ax_D1.axhline(speeds_arr.mean(), color=ORANGE, linestyle="--", linewidth=0.9, alpha=0.7,
              label=f"mean {speeds_arr.mean():.1f} m/s")
ax_D1.set_ylabel("Speed (m/s)", color=MUTED, fontsize=8)
ax_D1.set_title("D  D46 head-to-head — speed & position (full episode)", color="white", fontsize=10, pad=6)
ax_D1.legend(fontsize=7, facecolor=BG, edgecolor=SPINE, labelcolor="white", loc="lower right")
ax_D1.set_xticklabels([])
ax_D1.set_ylim(0, 32)

# track_gap subplot
# Shade regions: green when ego ahead (gap < 0), red when behind (gap > 0)
ax_D2.fill_between(steps_arr, gaps_arr, 0,
                   where=(gaps_arr < 0), color=GREEN,  alpha=0.3, label="Ego ahead")
ax_D2.fill_between(steps_arr, gaps_arr, 0,
                   where=(gaps_arr >= 0), color=RED,   alpha=0.3, label="Opponent ahead")
ax_D2.plot(steps_arr, gaps_arr, color=PURPLE, linewidth=0.8, zorder=3)
ax_D2.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
ax_D2.set_ylabel("track_gap (obs[11])", color=MUTED, fontsize=8)
ax_D2.set_xlabel("Step", color=MUTED, fontsize=8)
ax_D2.set_ylim(-1.05, 1.05)
ax_D2.legend(fontsize=7, facecolor=BG, edgecolor=SPINE, labelcolor="white", loc="upper right")

# Annotate fraction ahead
pct_ahead = (gaps_arr < 0).mean() * 100
ax_D2.text(
    0.02, 0.05, f"Ego ahead {pct_ahead:.0f}% of steps",
    transform=ax_D2.transAxes, color=GREEN, fontsize=8, va="bottom",
)

# Shared x-axis formatting for D sub-panels
for ax in [ax_D1, ax_D2]:
    ax.spines[:].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_facecolor(PANEL)

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = project_root / "plots" / "d46_analysis.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"[Viz] Saved → {save_path}")
