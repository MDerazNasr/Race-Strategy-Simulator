"""
Side-by-side Trajectory GIF — 4 policies racing on the same oval.

2×2 animated grid:
  Top-left:     Expert (rule-based, 17 m/s)
  Top-right:    cv2 — PPO speed champion (26.9 m/s)
  Bottom-left:  D37 — best pit policy (23.7 m/s, 3 pits shown as flash)
  Bottom-right: D46 — multi-agent champion (ego + opponent, 27.28 m/s)

Each panel shows:
  - Dashed oval track
  - Fading trail of the last N positions
  - Bright dot for the current car position
  - Live speed readout

D37: pit events flash the panel background red momentarily.
D46: two cars — ego (purple) and opponent (orange).

Saves to: plots/trajectory_comparison.gif
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env
from env.track import generate_oval_track
from expert.expert_driver import ExpertDriver
from env.car_model import DynamicCar
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — needed for PPO.load pickle
from stable_baselines3 import PPO

# ── Parameters ────────────────────────────────────────────────────────────────
N_LAPS       = 4       # record this many laps per policy
TRAIL_LEN    = 40      # how many past positions to show as a fading trail
FRAME_SKIP   = 3       # animate every Nth step (speeds up GIF)
FPS          = 30
device       = "cpu"

track = generate_oval_track()
print("[Viz] Collecting trajectories...")

# ── Collect trajectories ──────────────────────────────────────────────────────

def collect_trajectory(policy_fn, env, n_laps, fixed_start=True, extra_info_keys=None):
    """Run policy until n_laps completed, return list of frame dicts."""
    if extra_info_keys is None:
        extra_info_keys = []
    obs, _ = env.reset(options={"fixed_start": fixed_start} if fixed_start else {})
    frames = []
    laps_done = 0
    for _ in range(30_000):
        action, _ = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = {"x": env.car.x, "y": env.car.y, "speed": info["speed"]}
        for k in extra_info_keys:
            frame[k] = info.get(k, 0)
        frames.append(frame)
        if info["laps_completed"] > laps_done:
            laps_done = info["laps_completed"]
        if laps_done >= n_laps or terminated:
            break
    print(f"  → {len(frames)} steps, {laps_done} laps, final speed {frames[-1]['speed']:.1f} m/s")
    return frames

# 1. Expert
print("[Viz] Expert...")
env_expert = F1Env()
expert_driver = ExpertDriver(track, max_speed=17.0, lookahead=8, corner_factor=12.0, include_pit=False)
def expert_policy(obs):
    action = expert_driver.get_action(env_expert.car)
    return np.array(action[:2], dtype=np.float32), None
traj_expert = collect_trajectory(expert_policy, env_expert, N_LAPS)

# 2. cv2 — PPO speed champion
print("[Viz] cv2...")
model_cv2 = PPO.load(str(project_root / "rl" / "ppo_curriculum_v2.zip"), device=device)
env_cv2 = F1Env(multi_lap=True)
def cv2_policy(obs):
    action, _ = model_cv2.predict(obs, deterministic=True)
    return action, None
traj_cv2 = collect_trajectory(cv2_policy, env_cv2, N_LAPS)

# 3. D37 — best pit policy (show pit events)
print("[Viz] D37...")
model_d37 = PPO.load(str(project_root / "rl" / "ppo_pit_v4_d37.zip"), device=device)
env_d37 = F1Env(multi_lap=True, tyre_degradation=True, pit_stops=True,
                voluntary_pit_reward=True, voluntary_pit_threshold=0.60)
def d37_policy(obs):
    action, _ = model_d37.predict(obs, deterministic=True)
    return action, None
traj_d37 = collect_trajectory(d37_policy, env_d37, N_LAPS,
                               extra_info_keys=["pit_count", "tyre_life"])
# Mark pit event frames
pit_frames_d37 = set()
prev_pit = 0
for i, f in enumerate(traj_d37):
    if f["pit_count"] > prev_pit:
        for j in range(max(0, i-2), min(len(traj_d37), i+8)):
            pit_frames_d37.add(j)
        prev_pit = f["pit_count"]

# 4. D46 — multi-agent champion (ego + opponent, opp=25 m/s, pos_bonus=2.0)
print("[Viz] D46...")
model_d46 = PPO.load(str(project_root / "rl" / "ppo_multi_agent_d46.zip"), device=device)
from env.f1_multi_env import F1MultiAgentEnv
env_d46 = F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)

obs_d46, _ = env_d46.reset(options={"fixed_start": True})
traj_d46 = []
laps_d46 = 0
for _ in range(30_000):
    action, _ = model_d46.predict(obs_d46, deterministic=True)
    obs_d46, _, terminated, truncated, info_d46 = env_d46.step(action)
    traj_d46.append({
        "x": env_d46.car.x, "y": env_d46.car.y,
        "speed": info_d46["speed"],
        "opp_x": env_d46.opp_car.x, "opp_y": env_d46.opp_car.y,
        "opp_speed": info_d46.get("opp_speed", 25.0),
    })
    if info_d46["laps_completed"] > laps_d46:
        laps_d46 = info_d46["laps_completed"]
    if laps_d46 >= N_LAPS or terminated:
        break
print(f"  → {len(traj_d46)} steps, {laps_d46} laps")

# ── Cap all trajectories to the same length ───────────────────────────────────
max_len = min(len(traj_expert), len(traj_cv2), len(traj_d37), len(traj_d46))
traj_expert = traj_expert[:max_len]
traj_cv2    = traj_cv2[:max_len]
traj_d37    = traj_d37[:max_len]
traj_d46    = traj_d46[:max_len]

frame_indices = list(range(0, max_len, FRAME_SKIP))
print(f"[Viz] Animating {len(frame_indices)} frames ({max_len} steps at 1:{FRAME_SKIP})...")

# ── Set up figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor="#0D1117")
fig.suptitle("F1 Racing Agent — Policy Comparison", color="white",
             fontsize=14, fontweight="bold")

PANEL_CONFIGS = [
    {"title": "Expert (rule-based)",      "color": "#66BB6A", "traj": traj_expert},
    {"title": "cv2 — Speed Champion",     "color": "#42A5F5", "traj": traj_cv2},
    {"title": "D37 — Pit Strategy",       "color": "#FF7043", "traj": traj_d37},
    {"title": "D46 — Multi-Agent Champion", "color": "#AB47BC", "traj": traj_d46},
]

axs = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

# Static track elements
track_x = track[:, 0]
track_y = track[:, 1]
pad = 10

for ax, cfg in zip(axs, PANEL_CONFIGS):
    ax.set_facecolor("#0D1117")
    ax.plot(track_x, track_y, "--", color="#333333", linewidth=1.2, alpha=0.8)
    # Close the loop
    ax.plot([track_x[-1], track_x[0]], [track_y[-1], track_y[0]], "--",
            color="#333333", linewidth=1.2, alpha=0.8)
    ax.set_xlim(track_x.min() - pad, track_x.max() + pad)
    ax.set_ylim(track_y.min() - pad, track_y.max() + pad)
    ax.set_aspect("equal")
    ax.set_title(cfg["title"], color="white", fontsize=9.5, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_color("#30363D")

# Animated elements — one per panel
trails = []      # line objects for the fading trail
dots   = []      # scatter for current position
speed_texts = [] # text showing current speed
opp_trails = []  # D39 opponent trail
opp_dots   = []  # D39 opponent dot
tyre_texts  = [] # D37 tyre life text

for i, (ax, cfg) in enumerate(zip(axs, PANEL_CONFIGS)):
    trail, = ax.plot([], [], "-", color=cfg["color"], linewidth=1.0, alpha=0.5)
    dot,   = ax.plot([], [], "o", color=cfg["color"], markersize=6, zorder=5)
    trails.append(trail)
    dots.append(dot)
    speed_texts.append(ax.text(
        0.05, 0.95, "", transform=ax.transAxes,
        color="white", fontsize=8, va="top", ha="left",
    ))
    opp_trails.append(None)
    opp_dots.append(None)
    tyre_texts.append(None)

# D46: add opponent elements
opp_trail_d46, = axs[3].plot([], [], "-", color="#FF8F00", linewidth=0.8, alpha=0.4)
opp_dot_d46,   = axs[3].plot([], [], "o", color="#FF8F00", markersize=5, zorder=4)
opp_trails[3]  = opp_trail_d46
opp_dots[3]    = opp_dot_d46
axs[3].plot([], [], "-o", color=PANEL_CONFIGS[3]["color"], markersize=4,
            label="Ego (PPO)")
axs[3].plot([], [], "-o", color="#FF8F00", markersize=4, label="Opponent (25 m/s)")
axs[3].legend(loc="lower right", fontsize=7, facecolor="#161B22",
              edgecolor="#30363D", labelcolor="white")

# D37: tyre life text
tyre_texts[2] = axs[2].text(
    0.05, 0.05, "", transform=axs[2].transAxes,
    color="#81C784", fontsize=8, va="bottom", ha="left",
)

def update(frame_idx):
    artists = []
    trajs = [traj_expert, traj_cv2, traj_d37, traj_d46]

    for i, (trail, dot, stext, cfg, traj) in enumerate(
            zip(trails, dots, speed_texts, PANEL_CONFIGS, trajs)):

        fi = min(frame_idx, len(traj) - 1)
        start = max(0, fi - TRAIL_LEN)
        xs = [traj[j]["x"] for j in range(start, fi + 1)]
        ys = [traj[j]["y"] for j in range(start, fi + 1)]

        trail.set_data(xs, ys)
        dot.set_data([traj[fi]["x"]], [traj[fi]["y"]])
        stext.set_text(f"{traj[fi]['speed']:.1f} m/s")
        artists += [trail, dot, stext]

        # D39 opponent
        if i == 3 and opp_trails[3] is not None:
            oxs = [traj[j]["opp_x"] for j in range(start, fi + 1)]
            oys = [traj[j]["opp_y"] for j in range(start, fi + 1)]
            opp_trails[3].set_data(oxs, oys)
            opp_dots[3].set_data([traj[fi]["opp_x"]], [traj[fi]["opp_y"]])
            artists += [opp_trails[3], opp_dots[3]]

        # D37 pit flash + tyre display
        if i == 2:
            is_pit = frame_idx in pit_frames_d37
            axs[2].set_facecolor("#3a0a0a" if is_pit else "#0D1117")
            tl = traj[fi].get("tyre_life", 1.0)
            tyre_texts[2].set_text(f"Tyre: {tl:.2f}")
            artists.append(tyre_texts[2])

    return artists

ani = animation.FuncAnimation(
    fig, update, frames=frame_indices,
    interval=1000 // FPS, blit=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = project_root / "plots" / "trajectory_comparison.gif"
ani.save(str(save_path), writer="pillow", fps=FPS,
         savefig_kwargs={"facecolor": "#0D1117"})
print(f"[Viz] Saved → {save_path}")
