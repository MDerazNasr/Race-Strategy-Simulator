"""
Tyre Life Trace — D37's strategic pit timing in one episode.

Three-panel plot for a single fixed-start episode of D37 (best pit policy):
  Panel 1: Speed vs step — shows acceleration out of pits, tyre-wear slowdown
  Panel 2: Tyre life vs step — shows degradation, pit resets, voluntary timing
  Panel 3: Cumulative reward vs step — shows the payoff of each pit stop

Pit events marked as vertical dashed lines across all panels.
Lap completions marked as background ticks.

Saves to: plots/tyre_trace_d37.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — needed for PPO.load pickle

# ── Run one fixed-start episode ───────────────────────────────────────────────
device = "cpu"
model_path = str(project_root / "rl" / "ppo_pit_v4_d37.zip")
model = PPO.load(model_path, device=device)
print(f"[Viz] Loaded D37 from {model_path}")

env = F1Env(
    multi_lap=False,
    tyre_degradation=True,
    pit_stops=True,
    voluntary_pit_reward=True,
    voluntary_pit_threshold=0.60,
)
obs, _ = env.reset(options={"fixed_start": True})

steps, speeds, tyre_lives, cumulative_rewards = [], [], [], []
pit_steps, lap_steps = [], []
total_reward = 0.0

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    steps.append(step)
    speeds.append(info["speed"])
    tyre_lives.append(info["tyre_life"])
    cumulative_rewards.append(total_reward)

    if info["pit_count"] > len(pit_steps):
        pit_steps.append(step)

    if step > 0 and info["laps_completed"] > (steps[-2] // 200 if len(steps) > 1 else 0):
        pass  # use env tracking below

    if terminated or truncated:
        break

# Detect lap completions from tyre_life jumps and cumulative reward jumps
# Actually track them from env directly
env2 = F1Env(
    multi_lap=False, tyre_degradation=True, pit_stops=True,
    voluntary_pit_reward=True, voluntary_pit_threshold=0.60,
)
obs2, _ = env2.reset(options={"fixed_start": True})
prev_laps = 0
lap_steps = []
for step in range(len(steps)):
    action, _ = model.predict(obs2, deterministic=True)
    obs2, _, terminated2, truncated2, info2 = env2.step(action)
    if info2["laps_completed"] > prev_laps:
        lap_steps.append(step)
        prev_laps = info2["laps_completed"]
    if terminated2 or truncated2:
        break

steps_arr = np.array(steps)
speeds_arr = np.array(speeds)
tyre_arr   = np.array(tyre_lives)
rew_arr    = np.array(cumulative_rewards)

print(f"[Viz] Episode: {len(steps)} steps, {len(pit_steps)} pits at steps {pit_steps}, "
      f"{len(lap_steps)} laps, final reward={total_reward:.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#0D1117",
                          gridspec_kw={"height_ratios": [1, 1, 1.2]})
fig.suptitle(
    "D37 — Best Pit Policy: One Fixed-Start Episode\n"
    f"3 pit stops, {len(lap_steps)} laps completed, {total_reward:.0f} total reward",
    color="white", fontsize=13, fontweight="bold"
)

PIT_COLOR  = "#FF4081"
LAP_COLOR  = "#AAAAAA"
GRID_COLOR = "#1C2128"

for ax in axes:
    ax.set_facecolor("#0D1117")
    ax.spines[:].set_color("#30363D")
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)
    # Lap markers (subtle)
    for ls in lap_steps:
        ax.axvline(ls, color=LAP_COLOR, alpha=0.25, linewidth=0.8, zorder=1)
    # Pit markers
    for i, ps in enumerate(pit_steps):
        ax.axvline(ps, color=PIT_COLOR, alpha=0.9, linewidth=1.5, linestyle="--", zorder=2)
        if ax is axes[0]:
            ax.text(ps + 15, ax.get_ylim()[1] if hasattr(ax, '_ylim_set') else 30,
                    f"PIT {i+1}", color=PIT_COLOR, fontsize=8, va="top")

# Panel 1: Speed
ax1 = axes[0]
ax1.plot(steps_arr, speeds_arr, color="#4FC3F7", linewidth=1.0, zorder=3)
ax1.set_ylabel("Speed (m/s)", color="#AAAAAA", fontsize=10)
ax1.set_ylim(0, 32)
ax1.axhline(23.67, color="#FF8C00", linestyle=":", linewidth=1, alpha=0.6,
            label="D37 avg speed (23.67 m/s)")
ax1.legend(loc="lower right", fontsize=8, facecolor="#161B22",
           edgecolor="#30363D", labelcolor="white")
# Re-add pit labels now we know ylim
for i, ps in enumerate(pit_steps):
    ax1.text(ps + 20, 29, f"PIT {i+1}", color=PIT_COLOR, fontsize=8.5,
             fontweight="bold", va="top")
# Lap labels
for i, ls in enumerate(lap_steps):
    ax1.text(ls + 5, 1.5, f"Lap {i+1}", color=LAP_COLOR, fontsize=7, va="bottom")

# Panel 2: Tyre life
ax2 = axes[1]
ax2.fill_between(steps_arr, tyre_arr, alpha=0.25, color="#81C784")
ax2.plot(steps_arr, tyre_arr, color="#81C784", linewidth=1.2, zorder=3)
ax2.axhline(0.60, color="#FFB74D", linestyle=":", linewidth=1.2, alpha=0.8,
            label="Voluntary pit threshold (0.60)")
ax2.set_ylabel("Tyre life", color="#AAAAAA", fontsize=10)
ax2.set_ylim(-0.02, 1.05)
ax2.legend(loc="upper right", fontsize=8, facecolor="#161B22",
           edgecolor="#30363D", labelcolor="white")
# Mark pit trigger tyre_life
for ps in pit_steps:
    if ps < len(tyre_arr):
        tl = tyre_arr[ps]
        ax2.annotate(f"tl={tl:.2f}", xy=(ps, tl), xytext=(ps + 30, tl + 0.08),
                     fontsize=7.5, color=PIT_COLOR, ha="left",
                     arrowprops=dict(arrowstyle="->", color=PIT_COLOR, lw=0.7))

# Panel 3: Cumulative reward
ax3 = axes[2]
ax3.plot(steps_arr, rew_arr, color="#FF8C00", linewidth=1.2, zorder=3)
ax3.fill_between(steps_arr, rew_arr, alpha=0.15, color="#FF8C00")
ax3.set_ylabel("Cumulative reward", color="#AAAAAA", fontsize=10)
ax3.set_xlabel("Step", color="#AAAAAA", fontsize=10)
# Annotate pit cost dips
for ps in pit_steps:
    if ps < len(rew_arr):
        ax3.annotate("−200\n(pit cost)",
                     xy=(ps, rew_arr[ps]), xytext=(ps + 40, rew_arr[ps] - 200),
                     fontsize=7, color=PIT_COLOR,
                     arrowprops=dict(arrowstyle="->", color=PIT_COLOR, lw=0.7))

# Lap bonus ticks
for ls in lap_steps:
    if ls < len(rew_arr):
        ax3.plot(ls, rew_arr[ls], "^", color=LAP_COLOR, markersize=5, zorder=4, alpha=0.6)

# Final reward label
ax3.text(steps_arr[-1] - 50, rew_arr[-1] + 50, f"{total_reward:.0f}",
         color="#FF8C00", fontsize=9, fontweight="bold", ha="right")

# Shared x-axis formatting
for ax in axes[:-1]:
    ax.set_xticklabels([])
axes[-1].tick_params(axis="x", colors="#AAAAAA")

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = project_root / "plots" / "tyre_trace_d37.png"
plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="#0D1117")
print(f"[Viz] Saved → {save_path}")
