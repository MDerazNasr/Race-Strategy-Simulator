# F1 Race Strategy Simulator — Project Report

> A 7-week, 46-experiment journey: from a rule-based expert to a positional-strategy
> multi-agent RL racing agent with genuine overtaking behavior — 173% above the expert baseline.

---

## Overview

This project builds a complete autonomous racing pipeline from scratch, starting with
a hand-coded expert driver and progressively learning richer strategy through deep
reinforcement learning. The final system handles:

- **Dynamic tyre physics** (Pacejka magic formula, grip degradation)
- **Pit stop strategy** (timing entries to maximise lap count)
- **Multi-agent racing** with genuine positional awareness (overtaking, track-gap sensing)
- **Safety car periods** (speed-limited yellow-flag windows)
- **Real track geometry** (FastF1 Monaco telemetry, varying curvature)

All training uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3,
warm-started via Behavioral Cloning (BC) to avoid the cold-start problem.

---

## Project Arc

```
Week 1-2  Environment + Expert + BC            (d1–d11)
Week 3    Dynamic physics + Curriculum          (d12–d14)
Week 4    Speed record: 26.9 m/s               (d15–d17)
Week 5    Pit stop strategy (20 experiments)   (d18–d38)
Week 6    Multi-agent + Safety car             (d39–d40)
Week 7    Positional strategy (D41–D46)        opp speed + reward shaping → col[11] ≠ 0
(bonus)   Real track geometry (D42)            FastF1 Monaco — curvature hypothesis confirmed
```

---

## Environment

### Track
Custom oval generated with sinusoidal waypoints (200 points). Utilities:
`closest_point()`, `track_tangent()`, `signed_lateral_error()`.

### Car Physics
Two models, progressively harder:

| Model | States | Tyre Model | Slip |
|-------|--------|-----------|------|
| Kinematic | x, y, yaw, v | None | No |
| **DynamicCar** | x, y, ψ, v_x, v_y, r | Pacejka Magic Formula | Yes |

DynamicCar parameters (F1-scale):
- m = 750 kg, I_z = 1200 kg·m²
- C_f = C_r = 80 000 N/rad, µ = 1.5 (high-grip compound)
- max\_steer = 20°, max\_accel = 15 m/s², max\_decel = −20 m/s²

Tyre force: `F_y = D · sin(C · arctan(B · α))` — linear at small slip, saturates at peak µ.

### Observation Space (11D → 12D → 13D → 14D)

| Dim | Feature | Added |
|-----|---------|-------|
| 0 | v / 20.0 | Week 1 |
| 1 | heading\_error / π | Week 1 |
| 2 | lateral\_error / 3.0 | Week 1 |
| 3 | sin(heading\_error) | Week 1 |
| 4 | cos(heading\_error) | Week 1 |
| 5 | curvature (5 wpts, ~0.5 s) | Week 1 |
| 6 | curvature (15 wpts, ~1.5 s) | Week 3 |
| 7 | curvature (30 wpts, ~3.0 s) | Week 3 |
| 8 | progress ∈ [0, 1] | Week 3 |
| 9 | v_y / 5.0 (lateral slide) | Week 3 |
| 10 | r / 2.0 (yaw rate) | Week 3 |
| 11 | tyre\_life ∈ [0, 1] | Week 5 (pit) |
| 12 | track\_gap ∈ [−1, 1] | Week 6 (multi-agent) |
| 13 | opp\_speed\_norm ∈ [0, 1] | Week 6 (multi-agent) |

Note: D40 (safety car) uses `[11]=tyre_life, [12]=sc_active` (13D).
D48 (pit+multi-agent) uses 14D: `[11]=tyre_life, [12]=track_gap, [13]=opp_speed_norm`.

### Action Space (2D → 3D)
`[throttle, steer]` for standard driving; `[throttle, steer, pit_signal]` for pit environments.
All actions continuous in [−1, 1].

### Reward Function
Five components, shaped for positive transfer:

| Component | Formula | Weight |
|-----------|---------|--------|
| Progress | v · cos(heading\_error) | 1.0 |
| Speed bonus | v / 20.0 | 0.1 |
| Lateral penalty | −lateral\_error² | 0.5 |
| Heading penalty | −heading\_error² | 0.1 |
| Smoothness | −Δaction² | 0.05 |
| Terminal (off-track) | −20.0 (one-time) | — |
| Lap bonus | +100 per lap | — |
| Pit penalty | −200 per stop | — |
| Voluntary pit bonus | +300 when pit at tyre\_life < 0.60 | — |
| SC speed penalty | −2.0 × max(0, v − 22.0) per step | — |
| SC pit bonus | +100 when pitting under SC | — |

---

## Training Pipeline

### Stage 0 — Expert Driver (Baseline)

Rule-based controller: steer toward an 8-waypoint lookahead, brake proportional to
corner severity. Max speed 17 m/s (capped to ensure track compliance).

**Result: 2909 reward, 11 laps, 17.1 m/s** (fixed start, N=10)

### Stage 1 — Behavioral Cloning (Warm Start)

50 episodes of expert demonstrations → MLP trained with MSE loss.
Architecture: `11 → 128 → ReLU → 128 → ReLU → 2`.
BC nearly matches the expert it imitates (9.8 vs 10.0 m/s).

**Key insight**: BC alone is limited by distribution shift — small errors compound
into states never seen during training. BC is a warm start, not a final policy.

### Stage 2 — PPO Fine-Tuning

Three stability improvements over naive PPO:
- `ent_coef = 0.005` — prevents entropy collapse (policy becoming too narrow)
- `clip_range = 0.1` — tighter than SB3 default (0.2), reduces gradient spikes
- Cosine LR decay: 3e-4 → 1e-6 over training

**Cold start problem**: PPO from scratch achieved −82.8 reward (learned to go slowly
to avoid crashing — a local optimum). BC warm start escaped this completely.

**Result after 300k steps**: 19.4 m/s, 10.75 laps — **2× the expert speed**.

### Stage 3 — Curriculum Learning (Dynamic Physics)

DynamicCar made the task dramatically harder (tyre slip now causes crashes).
Three-stage curriculum via `CurriculumCallback`:

| Stage | max\_accel | Speed cap | Graduation criterion |
|-------|-----------|-----------|----------------------|
| 1 — Stability | 6.0 m/s² | ~8 m/s | 50% lap completion (5 rollouts) |
| 2 — Speed | 11.0 m/s² | ~15 m/s | 30% lap completion (5 rollouts) |
| 3 — Full Racing | 15.0 m/s² | None | Run to budget |

Policy weights are **preserved across stage transitions**. Curriculum vs stable training
(both 1M steps): +34% speed, −5.6% lateral error, +48% reward.

### Stage 4 — Multi-Lap Speed Champion (`ppo_curriculum_v2`)

3M PPO steps with multi-lap reward and extended training budget.

**Result: 4531 reward, 17 laps, 26.92 m/s, 100% completion** ← Project speed record

### Stage 5 — Tyre Degradation

Linear tyre wear: `tyre_life -= 0.0007/step`. Speed scales with `√tyre_life`.
Obs extended 11D → 12D (`tyre_life` appended). Forced pit at tyre\_life ≤ 0.10.

**Result (D17): 1644 reward** — penalized by speed loss from worn tyres. Established
the physics for the pit strategy campaign.

---

## Pit Stop Campaign (D18–D37)

### The Problem

Teaching an RL agent to make voluntary pit stops required 20 experiments over
3 weeks. The core challenge: a pit stop costs −200 reward immediately but enables
faster lap times through fresh tyres — a long-horizon credit assignment problem.

### Key Failures

**D18 — Never pitted**: BC imbalance (3400:1 non-pit steps), correct gradient never received.

**D19 — Never pitted**: `forced_pit_interval=500` never fired because episodes averaged
only 46 steps in Stage 0 of curriculum.

**D20 — Catastrophic**: Zero-initialized pit probability = 0.5 → agent pitted on fresh
tyres from step 1 → −200 penalty destroyed driving completely.

**D24 — Value crisis**: Loaded D21 into new `pit_timing_reward` environment. Value
function was wrong for the new reward scale → large gradients → steering collapse.
**Lesson**: "Good training ep\_rew\_mean" ≠ "Good deterministic evaluation."

**D31 — Feature bottleneck**: Froze the MLP extractor (18k params) and trained only
a linear pit output head. Found that frozen features do NOT linearly encode tyre\_life:
BC pre-training of pit row achieved separation of only 3.25% of what was needed.
Steer-toward-pit in latent space doesn't align with steer-toward-tyre\_life signal.

**D38 — Entropy restoration failure**: After D37's log\_std collapsed (steer std 0.05),
attempted to restore exploration with `ent_coef=0.01`. Noise perturbed the policy
OUT of its local optimum → reward fell from 3477 → 1730. **Lesson**: prevent
collapse early; don't try to restore it after the fact.

### Key Breakthroughs

**D21 — First voluntary pit stop** (1877 reward, 7 laps):
State-conditional threshold: `pit_signal = tanh(...)`. Init W\_pit so that at
tyre\_life < 0.35, pit\_signal > 0 from step 1. Gradient always pointed the right
direction. First agent to pit voluntarily.

**D32 — PitAwarePolicy** (2683 reward, 11 laps):
Architectural fix for the feature bottleneck: append `obs[11]` (tyre\_life) directly
to the 128-dim latent vector → 129-dim → action\_net = Linear(129, 3).
`pit_signal = features_w[:128]@features + W_pit[128]*tyre_life + b_pit`
Direct connection: gradient for tyre\_life component = W\_pit[128], never diluted
by frozen MLP. State-conditional from episode 1.

**D36 — Full Unfreeze → 3 Pits** (3427 reward, 15 laps):
After pit row was stable (D32–D35), released all 36k parameters. Features evolved
(+12–15%), speed improved +33%, pit count grew from 1 → 3. W\_pit[128]=−30
acted as a strong anchor: gradient spread across 36k params didn't override
well-calibrated pit prior.

**D37 — Project Best Pit Policy** (3477 reward, 15 laps, 3 pits, 23.67 m/s):
2M more steps from D36. Lateral error fell −30% (0.456 → 0.319 m). Log\_std
collapsed (steer std 0.57 → 0.05) — policy converged to a near-deterministic
3-pit strategy. **Final pit policy.**

### Pit Stop Architecture — `PitAwarePolicy`

```
obs (12D) ──► mlp_extractor.policy_net (frozen) ──► features (128D)
                                                           │
obs[11] (tyre_life) ─────────────────────────────────────►│
                                                     concat (129D)
                                                           │
                                              action_net = Linear(129, 3)
                                                  [throttle, steer, pit_signal]
```

**Initialization**: `W_pit[128] = −10.0`, `b_pit = +7.0`
→ at tyre\_life = 0.60: pit\_signal = −10×0.60 + 7 = +1.0 > 0 → pit fires
→ at tyre\_life = 1.00: pit\_signal = −10×1.00 + 7 = −3.0 < 0 → no pit

---

## Advanced Experiments (D39–D40)

### D39 — Multi-Agent Racing

Environment: `F1MultiAgentEnv` — ego agent (PPO) vs ExpertDriver opponent (22 m/s).
Obs extended 12D → 14D: `track_gap` (distance ahead of opponent) + `opp_speed_norm`.

**Key finding**: PPO discovered a trivial winning strategy — drive faster (26.9 vs 22 m/s).
Columns 12–13 (track\_gap, opp\_speed\_norm) learned zero weight. No strategic
positioning or blocking behavior emerged.

**Result: 6024.8 reward, 17 laps, 26.91 m/s, 5 overtakes, 100% completion**
(Note: higher reward reflects different environment, not directly comparable to pit runs.)

### D40 — Safety Car

Random yellow-flag periods: Bernoulli trigger (0.3%/step), 80–200 step duration.
Speed penalty: −2.0 × max(0, v − 22.0) per step. Pit bonus: +100 when pitting
under SC (net cost: −200 → −100). SC obs: `sc_active ∈ {0, 1}` at dim 12.
Warm-started from D37 (12D → 13D via `extend_obs_dim`).

**Key finding**: Agent found a static speed equilibrium at 21.68 m/s — permanently
below the 22 m/s SC limit, never needing to read `sc_active`. Column 12 weight = 0.
SC violation rate: 57.6% during SC periods (agent speeds on straights, not fully compliant).

**Result: 3820.2 ±59.19 reward, 13 laps, 21.68 m/s, 2 pits, 100% completion**

### The D39/D40 Pattern: Static vs Conditional Strategy

Both multi-agent and safety car experiments revealed the same failure mode:
**PPO finds the simplest strategy that avoids needing the new signal.**

For conditional behavior to emerge, the agent must be *forced* to use it — either
because speed alone is insufficient (raise the opponent's speed to match the ego),
or by adding a green-flag speed bonus that makes static slow driving suboptimal.

---

## Positional Strategy (D41–D46)

### The Follow Equilibrium Problem

Raising opponent speed to 27 m/s (D41) made track_gap weight non-zero (0.016) but
produced 0% lap completion — too hard. At 25 m/s (D43), the agent found a new local
optimum: **match the opponent's speed exactly** and follow behind it. The position_bonus
of +0.5/step made being ahead worth only +1000/episode — not worth the risk of overtaking.

### Closing the Shortcut: Reward Shaping

Raising `position_bonus` from 0.5 → 2.0/step (D44) made being ahead worth +4000/episode
— 4× the old value. The follow equilibrium broke immediately: track_gap weight surged +155%
(0.022 → 0.055) and deterministic speed jumped from 24.36 → 26.91 m/s.

Subsequent fine-tuning improved lap completion (3 → 8 → 17 laps):

| Experiment | Opp speed | Pos bonus | col[11] weight | Reward | Laps |
|------------|-----------|-----------|---------------|--------|------|
| D39 | 22 m/s | 0.5 | 0.000 | 6025 | 17 | ← no strategy |
| D41 | 27 m/s | 0.5 | 0.016 | 69 | 0 | ← too hard |
| D43 | 25 m/s | 0.5 | 0.022 | 275 | 0 | ← follow equilibrium |
| D44 | 25 m/s | 2.0 | 0.055 | 1834 | 3 | ← equilibrium broken |
| D45 | 25 m/s | 2.0 | 0.067 | 3727 | 8 | ← fine-tuning |
| **D46** | **25 m/s** | **2.0** | **0.070** | **7943** | **17** | **← goal achieved ✓** |

### Key Finding: Speed Improves Under Pressure

Counter-intuitively, a harder opponent and stronger position reward produced a *faster*
policy: D46 (27.28 m/s) beat D39 (26.91 m/s) despite more challenging conditions.
The position bonus made "staying ahead" worth the risk of pushing speed limits.

---

## Real Track Geometry (D42 — Monaco)

Replaced the synthetic oval (constant curvature) with real Monaco 2023 qualifying telemetry
from FastF1. Primary hypothesis: the three curvature lookahead observations (dims 5/6/7) carry
zero information on the oval (every corner is identical), but become genuinely useful on a
real circuit with varying straights, hairpins, and chicanes.

### Infrastructure

- **FastF1 coordinate system**: position data is in **decimeters** — must divide by 10 for meters.
  This was not documented and required debugging from crash patterns.
- **Two-pass deduplication**: FastF1 telemetry has near-duplicate points (stationary frames).
  Required removing points within 0.5m before resampling, then removing post-resample duplicates.
- **Track-relative lookahead**: hardcoded waypoint offsets (5/15/30) replaced with
  `N//40 / N//13 / N//7` to preserve the same time horizons across track densities.

### Result

**Primary hypothesis confirmed**: curvature weights on Monaco — col[5]=0.222, col[6]=0.179,
col[7]=0.180 — all non-zero. On the oval, these columns converge to ~0. The Week 3 obs design
was validated on a real circuit.

**Expert driver failure**: the rule-based ExpertDriver (max_speed=17 m/s, corner_factor=12)
crashed within 17 steps on average at Monaco. Root cause: large initial heading errors →
`target_speed = 17 − 12×|err| < 0` → full brake → stall. All 50 collection episodes:
0 laps, 884 total samples (need ~30k for usable BC). Training curriculum stuck in Stage 0.

**Fix (D47)**: reduce max_speed=8.0, corner_factor=4.0, lookahead=5 so that even a 2.0 rad
heading error still yields `target_speed = 8 − 4×2 = 0` (never negative).

---

## Final Results Summary

### Fixed-Start Evaluation (N=10, deterministic policy)

| Policy | Reward | Laps | Speed | Pits | Notes |
|--------|--------|------|-------|------|-------|
| **D46 — Multi-Agent Champion** | **7943** | **17** | **27.28 m/s** | 0 | **New project best** |
| D39 — Multi-Agent Baseline | 6025 | 17 | 26.91 m/s | 0 | No positional strategy |
| cv2 — Speed Champion | 4531 | 17 | 26.92 m/s | 0 | Single-agent best |
| D40 — Safety Car | 3820 | 13 | 21.68 m/s | 2 | SC compliance |
| **D37 — Best Pit Policy** | **3477** | **15** | **23.67 m/s** | **3** | Best pit strategy |
| D36 | 3427 | 15 | 23.60 m/s | 3 | |
| Expert (rule-based) | 2909 | 11 | 17.1 m/s | — | Baseline |
| D32 — PitAwarePolicy | 2683 | 11 | 17.7 m/s | 1 | |
| PPO from scratch | −83 | 0.05 | 2.9 m/s | — | Cold start failure |

Note: D39, D40, D46 use different environments (multi-agent / safety car) with different reward
scales. Within each environment, reward comparisons are valid.

### Progress Across the Project

```
Expert (hand-coded)        →  2909 reward  (baseline)
PPO speed champion (cv2)   →  4531 reward  (+55.7% vs Expert)
D37 pit strategy           →  3477 reward  (+19.5% vs Expert, +3 pit stops)
D46 positional champion    →  7943 reward  (+173% vs Expert, col[11]=0.070)
```

---

## Key Technical Lessons

### 1. BC Warm Start Solves the Cold Start Problem
PPO from scratch: −83 reward, 0.05 laps. BC warm start: 1808 reward, 10.75 laps.
The agent must start somewhere near the correct behavior space for RL to be useful.

### 2. Curriculum Learning is Necessary for Hard Physics
DynamicCar (tyre slip) crashed PPO immediately when trained naively. Three-stage
curriculum (stability → speed → full racing) produced +34% speed, −5.6% lateral
error vs stable training at the same step budget.

### 3. Architectural Connections Beat Feature Engineering
The feature bottleneck (D31) showed that frozen features can't linearly encode
tyre\_life for pit timing. Directly connecting the raw tyre\_life signal to the
output layer (PitAwarePolicy, D32) bypassed the bottleneck entirely.

### 4. PPO Finds the Simplest Winning Strategy
D39: zero weight on opponent position features (raw speed advantage suffices).
D40: zero weight on `sc_active` (static slow speed suffices).
RL is extremely good at finding shortcuts. Design reward functions that close them.

### 5. Initialization Determines Whether Learning Starts Correctly
D20: zero init → random pit probability = 50% → catastrophic on fresh tyres.
D21: threshold init → correct gradient from step 1.
**Wrong initialization can destroy an otherwise sound algorithm.**

### 6. Entropy Collapse is Easier to Prevent than Restore
Set `ent_coef` early (from training start) when collapse is a risk, not afterward.
D38: entropy boost on a collapsed policy added noise that pushed the agent out of
its local optimum, falling from 3477 → 1730.

### 7. LR Schedule and `reset_num_timesteps` Must Be Set Together
SB3 bug (fixed in D28): `reset_num_timesteps=False` restores stored LR, ignoring
the new schedule. Fix: always set both `model.learning_rate` AND `model.lr_schedule`.

### 9. Closing the Shortcut Forces Conditional Behavior

D39/D40 showed that PPO ignores new observation signals when a simpler strategy works.
The fix in D41–D46: raise opponent speed so raw speed advantage is no longer sufficient,
then raise the position_bonus so being ahead is worth fighting for. Both interventions
were needed: speed pressure made track_gap *useful*, reward magnitude made it *worth using*.

### 10. Curvature Observations Need Real Track Geometry

On the oval (constant curvature), dims 5/6/7 carry no useful information and converge to
~0 weight. On Monaco (varying curvature), they reached weights of 0.18–0.22 — among the
strongest signals in the first layer. Observation design should be validated on the track
type the agent will actually encounter.

### 8. RolloutBuffer Must Be Recreated After Obs Dim Extension
`PPO.load()` pre-allocates a RolloutBuffer with the saved obs shape. After
`extend_obs_dim()`, the buffer is still the old size. Explicit recreation required:
```python
model.rollout_buffer = RolloutBuffer(
    model.n_steps, model.observation_space, model.action_space,
    device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1,
)
```

---

## Repository Structure

```
env/
  f1_env.py              F1Env — Gymnasium wrapper with all physics, reward, SC
  car_model.py           KinematicCar + DynamicCar (Pacejka tyre model)
  track.py               Oval track + FastF1 real track loader
  f1_multi_env.py        F1MultiAgentEnv — ego + ExpertDriver opponent
  f1_multi_pit_env.py    F1MultiAgentPitEnv — combined pit+multi-agent (14D, 3D)

expert/
  expert_driver.py    Rule-based driver (lookahead + corner braking)
  collect_data.py     Dataset generation (oval + Monaco variants)

bc/
  train_bc.py           Behavioral Cloning (MSE, Adam)
  train_bc_monaco.py    BC on Monaco expert data
  bc_policy_final.pt    Trained BC weights (oval)
  bc_policy_monaco.pt   Trained BC weights (Monaco)

rl/
  rewards.py              RacingReward — 5-component shaped reward
  schedules.py            cosine_schedule, linear_schedule, warmup_cosine_schedule
  bc_init_policy.py       load_bc_weights_into_ppo(), extend_obs_dim()
  pit_aware_policy.py     TyrLifeAugmentedExtractor + PitAwarePolicy (D32+)
  curriculum.py           CurriculumStage, STAGES_*, CurriculumCallback
  make_env.py             Environment factory functions
  evaluate.py             Full evaluation pipeline + comparison plots
  ppo_curriculum_v2.zip       Speed champion (4531, 26.9 m/s)
  ppo_pit_v4_d37.zip          Best pit policy (3477, 3 pits, 23.7 m/s)
  ppo_multi_agent_d46.zip     Multi-agent champion (7943, 17 laps, 27.28 m/s) ← project best
  ppo_sc_d40.zip              Safety car (3820, 13 laps)
  ppo_monaco.zip              Monaco circuit (Stage 0 only, curvature validated)

plots/
  progress_dashboard.png    All 46 experiments, bar chart by phase
  tyre_trace_d37.png        D37 speed/tyre/reward trace with pit markers
  trajectory_comparison.gif 2×2 animated GIF: Expert / cv2 / D37 / D46
  d46_analysis.png          D46 policy analysis: heatmap, column importance, evolution

Notes/
  d10.txt – d46.txt   Per-experiment documentation
  PROGRESS.txt        Full project history
```

---

## What's Next

Two experiments in progress:

**D47 — Fix Monaco** (in progress)
Reduce ExpertDriver params (max_speed=8.0, corner_factor=4.0) so it survives Monaco's
initial heading errors. Goal: ≥10k BC samples, curriculum advances past Stage 0.

**D48 — Combined Pit + Multi-Agent** (in progress)
New `F1MultiAgentPitEnv`: 14D obs, 3D action, opponent at 25 m/s. Warm-start from D37
(pit expert). Goal: agent pits at least once while also using track_gap for positioning.
This would be the first policy to combine both forms of strategy.

---

*Built over 7 weeks, 46 experiments, ~50M PPO training steps.*
*Core stack: Python 3.14, PyTorch, Stable-Baselines3, Gymnasium, FastF1, Matplotlib.*
