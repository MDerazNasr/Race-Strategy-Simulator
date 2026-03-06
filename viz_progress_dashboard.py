"""
Progress Dashboard — full project arc in one figure.

Shows fixed-start reward for every experiment (d10–d40), annotated with
key breakthroughs. Horizontal reference lines for Expert and cv2 baselines.
Color-coded by phase: single-agent driving, pit strategy experiments, advanced.

Saves to: plots/progress_dashboard.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data: (label, day_num, reward, phase) ────────────────────────────────────
# phase: 0 = driving foundation, 1 = pit stop journey, 2 = advanced (multi/SC)
# Reward = fixed-start deterministic eval (N=10) from evaluate.py
EXPERIMENTS = [
    # --- Driving foundation (d10–d17) ---
    ("BC\n(imitation)",      10,  2909, 0),
    ("PPO+BC\n(stable)",     11,   135, 0),
    ("PPO\ncurriculum",      14,   259, 0),
    ("cv2\n(3M steps)",      16,  4531, 0),   # curriculum v2 — the speed champion
    ("Multi-lap\n(3M+)",     16.5, -10, 0),   # regression from multi-lap
    ("PPO\ntyre",            17,  1644, 0),

    # --- Pit stop journey (d18–d38) ---
    ("d18\n(first pit)",     18,   942, 1),
    ("d19",                  19,   827, 1),
    ("d20\n(catastrophe)",   20,   -18, 1),
    ("d21\n★ first pit!",    21,  1877, 1),
    ("d22",                  22,  2283, 1),
    ("d23",                  23,  2174, 1),
    ("d24",                  24,    -2, 1),
    ("d25",                  25,  1276, 1),
    ("d26",                  26,  1883, 1),
    ("d27",                  27,  1883, 1),
    ("d28",                  28,  1883, 1),
    ("d29",                  29,  1885, 1),
    ("d30",                  30,  1880, 1),
    ("d31",                  31,  1880, 1),
    ("d32\n★ PitAware",      32,  2684, 1),
    ("d33",                  33,  2684, 1),
    ("d34",                  34,  2689, 1),
    ("d35",                  35,  2688, 1),
    ("d36\n★ 3 pits!",       36,  3427, 1),
    ("d37\n(best pit)",      37,  3477, 1),
    ("d38\n(failed)",        38,  1731, 1),

    # --- Advanced (d39–d40) ---
    ("d39\nmulti-agent *",   39,  6025, 2),
    ("d40\nsafety car *",    40,  3820, 2),
]

PHASE_COLORS = {
    0: "#4FC3F7",   # light blue — driving foundation
    1: "#FFB74D",   # orange — pit strategy
    2: "#81C784",   # green — advanced
}
PHASE_LABELS = {
    0: "Driving foundation",
    1: "Pit stop experiments (20 runs)",
    2: "Advanced: multi-agent / safety car\n(* different env, reward not directly comparable)",
}

EXPERT_REWARD = 2909
CV2_REWARD    = 4531

# ── Build figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7), facecolor="#0D1117")
ax.set_facecolor("#0D1117")

labels   = [e[0] for e in EXPERIMENTS]
x_pos    = list(range(len(EXPERIMENTS)))
rewards  = [e[2] for e in EXPERIMENTS]
phases   = [e[3] for e in EXPERIMENTS]
colors   = [PHASE_COLORS[p] for p in phases]

# Bar chart
bars = ax.bar(x_pos, rewards, color=colors, alpha=0.85, width=0.7, zorder=3)

# Highlight negative bars with a red tint
for bar, r in zip(bars, rewards):
    if r < 0:
        bar.set_color("#EF5350")
        bar.set_alpha(0.9)

# Reference lines
ax.axhline(EXPERT_REWARD, color="#AAAAAA", linestyle="--", linewidth=1.2, zorder=2,
           label=f"Expert baseline ({EXPERT_REWARD})")
ax.axhline(CV2_REWARD, color="#FF8C00", linestyle="--", linewidth=1.5, zorder=2,
           label=f"cv2 speed champion ({CV2_REWARD})")
ax.axhline(0, color="#555555", linestyle=":", linewidth=0.8, zorder=2)

# Annotations for key breakthroughs
ANNOTATIONS = {
    "d21\n★ first pit!":   ("First voluntary\npit stop!", -300),
    "d32\n★ PitAware":     ("PitAwarePolicy:\ndirect tyre→pit", +150),
    "d36\n★ 3 pits!":      ("Full unfreeze:\n3 pits, 15 laps", +150),
    "cv2\n(3M steps)":     ("Speed champion:\n26.92 m/s", +150),
    "d39\nmulti-agent *":  ("Multi-agent:\n5 overtakes", +150),
}
for i, label in enumerate(labels):
    if label in ANNOTATIONS:
        text, dy = ANNOTATIONS[label]
        y = rewards[i]
        ax.annotate(
            text,
            xy=(i, y),
            xytext=(i, y + dy + (400 if dy > 0 else -400)),
            fontsize=7.5, color="white", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8),
        )

# Axes
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=7, color="#CCCCCC", rotation=0)
ax.set_ylabel("Fixed-start reward (N=10)", color="#AAAAAA", fontsize=11)
ax.tick_params(axis="y", colors="#AAAAAA")
ax.spines[:].set_color("#30363D")

ax.set_title(
    "F1 Racing Agent — Full Project Arc (40 Experiments)\n"
    "Fixed-start evaluation reward across all training runs",
    color="white", fontsize=13, fontweight="bold", pad=14,
)

# Phase shading
phase_boundaries = []
for i in range(1, len(phases)):
    if phases[i] != phases[i - 1]:
        phase_boundaries.append(i - 0.5)

shade_ranges = []
start = 0
for b in phase_boundaries + [len(phases)]:
    end = int(b + 0.5) if b != len(phases) else len(phases)
    shade_ranges.append((start, end - 1, phases[start]))
    start = int(b + 0.5)

for (s, e, p) in shade_ranges:
    ax.axvspan(s - 0.5, e + 0.5,
               color=PHASE_COLORS[p], alpha=0.06, zorder=1)

# Legend
legend_patches = [
    mpatches.Patch(color=PHASE_COLORS[p], alpha=0.85, label=PHASE_LABELS[p])
    for p in sorted(PHASE_COLORS)
]
legend_patches += [
    plt.Line2D([0], [0], color="#AAAAAA", linestyle="--", label=f"Expert ({EXPERT_REWARD})"),
    plt.Line2D([0], [0], color="#FF8C00", linestyle="--", label=f"cv2 ({CV2_REWARD})"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
          facecolor="#161B22", edgecolor="#30363D", labelcolor="white")

plt.tight_layout()
save_path = Path(__file__).parent / "plots" / "progress_dashboard.png"
plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="#0D1117")
print(f"[Viz] Saved → {save_path}")
