# F1 Race Strategy Simulator

A 6-week deep reinforcement learning project that builds an autonomous racing agent from scratch — starting with a hand-coded rule-based driver and progressively adding dynamic tyre physics, pit stop strategy, multi-agent racing, and safety car events.

**Final result: a pit-stopping agent that completes 15 laps with 3 timed pit stops, achieving 55% more reward than the expert it was trained on.**

---

## What We Built

The project follows a standard robotics pipeline, adding one layer of complexity at a time:

```
Expert driver (rules)
    ↓
Behavioral Cloning (imitation learning)
    ↓
PPO fine-tuning (RL surpasses the expert)
    ↓
Dynamic tyre physics (Pacejka model)
    ↓
Curriculum learning (3-stage difficulty ramp)
    ↓
Tyre degradation + pit stops (20-experiment campaign)
    ↓
Multi-agent racing (ego vs. opponent)
    ↓
Safety car / yellow flag events
```

40 experiments total across 6 weeks, ~30M PPO training steps.

---

## Results

### Policy Comparison (fixed start, 10 episodes, deterministic)

| Policy | Reward | Laps | Speed | Pit Stops |
|--------|--------|------|-------|-----------|
| cv2 — Speed Champion | 4531 | 17 | 26.9 m/s | 0 |
| D37 — Best Pit Policy | **3477** | **15** | **23.7 m/s** | **3** |
| Expert (rule-based) | 2909 | 11 | 17.1 m/s | — |
| PPO from scratch | −83 | 0 | 2.9 m/s | — |

The speed champion (`ppo_curriculum_v2`) achieves **26.9 m/s** — 58% faster than the expert it was initialised from. The pit policy (`D37`) completes 15 laps with 3 strategically timed pit stops despite a −200 reward cost per stop.

---

## Visualizations

### Policy Progress — All 40 Experiments

![Progress Dashboard](plots/progress_dashboard.png)

Every experiment from D10 to D40, color-coded by phase:
- **Blue** — driving foundation (BC → PPO → curriculum)
- **Orange** — pit stop campaign (20 experiments to get 3 voluntary pits)
- **Green** — advanced: multi-agent and safety car

Key moments: first voluntary pit stop (D21), PitAwarePolicy architectural fix (D32), full unfreeze → 3 pits (D36), speed record at 26.9 m/s (cv2).

---

### D37 — Pit Strategy Trace (One Episode)

![Tyre Trace](plots/tyre_trace_d37.png)

Three panels from a single fixed-start episode of the best pit policy:
- **Top**: Speed over time — acceleration out of each pit, tyre-wear slowdown between stops
- **Middle**: Tyre life — degradation curve resets to 1.0 at each pit entry
- **Bottom**: Cumulative reward — −200 dips at each pit, recovered by faster lap times

Three pit stops at steps 520, 1037, 1554. 15 laps completed. 4377 total reward.

---

### 4-Policy Trajectory Comparison (Animated)

![Trajectory GIF](plots/trajectory_comparison.gif)

Four agents racing simultaneously on the same oval:
- **Top-left (green)** — Expert driver: slow and steady, 17 m/s
- **Top-right (blue)** — cv2 speed champion: fastest lap times, 26.9 m/s, no pits
- **Bottom-left (orange)** — D37 pit strategy: panel flashes red on each pit stop
- **Bottom-right (purple + orange)** — D39 multi-agent: ego (purple) vs ExpertDriver opponent (orange)

---

## Key Technical Findings

### 1. Cold Start Problem
PPO from scratch explores randomly, crashes constantly, and converges to "go slowly to avoid crashing" — reward: −83. Behavioral Cloning warm start escaped this entirely: reward jumped to 1808 from the first episode of RL training.

### 2. Curriculum Learning is Necessary for Hard Physics
The Pacejka tyre model introduces lateral slip that crashes naive PPO immediately. A 3-stage curriculum (stability → speed → full racing) produced +34% speed and −5.6% lateral error vs. stable training at the same step budget.

### 3. The Feature Bottleneck (D31 — Critical Finding)
For pit stop timing, we tried training only a linear output head on top of frozen MLP features. Discovered that frozen features do not linearly encode tyre life — separation was only 3.25% of what was needed. The fix: **directly connect the raw tyre_life signal to the output layer** (PitAwarePolicy, D32), bypassing the bottleneck entirely.

### 4. PitAwarePolicy Architecture
```
obs (12D) ──► MLP extractor (frozen) ──► features (128D) ──► concat (129D) ──► action_net
                                                                  ▲
                                          obs[11] (tyre_life) ───┘
```
Direct connection means gradient for pit timing = W_pit × tyre_life — never diluted by the frozen feature layers. State-conditional pitting from training step 1.

### 5. PPO Finds the Simplest Winning Strategy
Both the multi-agent (D39) and safety car (D40) experiments revealed the same pattern: PPO ignores new observation signals if a simpler strategy already works. D39 used raw speed advantage (26.9 > 22 m/s) instead of positional strategy. D40 settled at 21.68 m/s (below the 22 m/s SC limit) instead of learning conditional fast/slow behavior. Learned weights on both new observation columns were exactly 0.

### 6. Entropy Collapse: Prevent, Don't Restore
After D37's policy converged (steer std 0.05), applying `ent_coef=0.01` hoping to restore exploration instead pushed the policy out of its local optimum — reward fell from 3477 → 1730. Set entropy regularization from the start; restoring it after collapse doesn't work.

---

## Architecture

**Car model**: 6-state DynamicCar — position (x, y), heading ψ, longitudinal velocity v_x, lateral velocity v_y, yaw rate r. Tyre forces via Pacejka Magic Formula (`F_y = D·sin(C·arctan(B·α))`).

**Observation (12D)**:
`[speed, heading_error, lateral_error, sin(hdg), cos(hdg), curv_near, curv_mid, curv_far, progress, v_y, yaw_rate, tyre_life]`

**Policy**: SB3 PPO with `MlpPolicy`, net_arch `[128, 128]`. Warm-started from BC weights. PitAwarePolicy for pit-stop experiments (129-dim action head).

**Training**: Cosine LR decay (1e-4 → 1e-6), `ent_coef=0.005–0.01`, `clip_range=0.1`. 1–3M steps per experiment.

---

## Repository Structure

```
env/
  f1_env.py            Gymnasium environment (tyre degradation, pit stops, safety car)
  car_model.py         KinematicCar + DynamicCar (Pacejka tyre model)
  track.py             Oval track + geometry utilities
  f1_multi_env.py      Multi-agent environment

expert/
  expert_driver.py     Rule-based driver (lookahead steering + corner braking)

bc/
  train_bc.py          Behavioral Cloning training

rl/
  pit_aware_policy.py  PitAwarePolicy — direct tyre_life → pit signal connection
  curriculum.py        3-stage CurriculumCallback
  evaluate.py          Evaluation pipeline + plots
  make_env.py          Environment factory functions
  ppo_curriculum_v2.zip       Speed champion model
  ppo_pit_v4_d37.zip          Best pit strategy model
  ppo_multi_agent_d39.zip     Multi-agent model
  ppo_sc_d40.zip              Safety car model

plots/
  progress_dashboard.png      All 40 experiments bar chart
  tyre_trace_d37.png          D37 tyre strategy trace
  trajectory_comparison.gif   4-policy animated comparison

viz_progress_dashboard.py    Reproduce progress_dashboard.png
viz_tyre_trace.py            Reproduce tyre_trace_d37.png
viz_trajectory_gif.py        Reproduce trajectory_comparison.gif

Notes/
  d10.txt – d40.txt    Per-experiment technical documentation
  PROGRESS.txt         Full project history

REPORT.md              Comprehensive technical report
```

---

## Stack

- Python 3.14
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) — environment interface
- PyTorch — neural network backend
- Matplotlib / Pillow — visualization and GIF generation
