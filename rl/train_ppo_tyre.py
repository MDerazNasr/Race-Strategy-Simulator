"""
PPO Tyre Degradation Training (Week 5 / d17).

WHAT IS THIS?
=============
This script continues from ppo_curriculum_v2.zip and fine-tunes the policy
to handle tyre degradation — a realistic F1 physics feature where grip
degrades every lap based on how hard the tyres are being used.

WHAT'S DIFFERENT FROM PREVIOUS TRAINING
========================================
All previous runs (d13 curriculum, d15 continued, d16 multi-lap) used
perfect grip: mu = 1.5 forever, every lap identical.

Here, DynamicCar degrades its own grip every step:
  wear = 0.0003 + 0.002 * (|alpha_f| + |alpha_r|)
  tyre_life -= wear
  mu = 1.5 * max(0.1, tyre_life)

At typical cornering (alpha_f + alpha_r ≈ 0.2 rad):
  wear ≈ 0.0007/step → tyres fully worn at ~1428 steps (~12 laps)

At aggressive cornering (alpha_f + alpha_r ≈ 0.5 rad):
  wear ≈ 0.0013/step → tyres fully worn at ~769 steps (~6 laps)

This creates a genuine trade-off:
  Push hard  → faster per lap,   but tyres die in 6 laps  → episode ends early
  Go easy    → slower per lap,   but tyres last 12 laps   → more total reward

The agent must discover the optimal pace that maximises TOTAL reward,
not just instantaneous speed.  This is the core of real F1 race strategy.

WHY NOT CATASTROPHIC FORGETTING (unlike d16)?
=============================================
d16 failed because it changed the EPISODE TERMINATION CONDITION.
The critic was calibrated for 2000-step episodes, but multi_lap = True
meant episodes could be infinite.  TD error became ~2000 units → collapse.

This run does NOT change termination:
  - Episodes still truncate at max_steps=2000.
  - Only the tyre physics change (grip degrades, episodes end a bit earlier).
  - The value function's calibration horizon (2000 steps) is still valid.

Additionally, the 12th observation dimension (tyre_life) starts with
ZERO WEIGHT in both actor and critic (extended via extend_obs_dim()).
This means the policy behaves EXACTLY like v2 on the first training step —
no sudden distribution shift.  The network gradually learns to use tyre_life
as training progresses.

OBSERVATION CHANGE: 11D → 12D
==============================
Old obs (11D, all previous training):
  [0-10] speed, heading, lateral, sin/cos, curvatures, progress, v_y, r

New obs (12D, tyre env):
  [0-10] same as before
  [11]   tyre_life ∈ [0, 1]  (1.0 = fresh, 0.0 = fully worn)

HOW THE 11D → 12D WEIGHT TRANSFER WORKS
=========================================
The first Linear layer in both actor and critic has shape (128, 11).
extend_obs_dim() creates a new (128, 12) weight by:
  1. Copying the old (128, 11) values into columns [:, :11]
  2. Setting column [:, 11] to zero

Effect: the policy produces IDENTICAL outputs to v2 on the first forward
pass (zero weight on new dim = new input has no effect yet).
Over 2M steps of training, the network learns what to do with tyre_life.

WHY CONTINUE FROM V2 INSTEAD OF TRAINING FROM SCRATCH?
=======================================================
ppo_curriculum_v2 already knows:
  - How to follow the track at 27 m/s
  - How to manage tyre slip (v_y, r) for sustained periods
  - How to complete sustained laps (100% lap completion, 17 laps/episode)

Starting from scratch would waste 3M steps re-learning basic driving.
The tyre degradation is a NEW challenge layered on top — fine-tuning is
the right tool.  The zero-weight extension ensures the starting point is
identical to v2, making catastrophic forgetting extremely unlikely.

SAVES TO: rl/ppo_tyre.zip
LOGS TO:  runs/ppo_tyre/
"""

import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_tyre
from rl.schedules import cosine_schedule
from rl.bc_init_policy import extend_obs_dim


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    ADDITIONAL_STEPS = 2_000_000

    # ── Tyre degradation environment (12D obs) ───────────────────────────────
    # make_env_tyre() returns F1Env(tyre_degradation=True) wrapped in Monitor.
    # Episodes end at max_steps=2000 (same as all previous training) — NOT
    # multi_lap.  This avoids the catastrophic forgetting failure from d16.
    env = DummyVecEnv([make_env_tyre])

    # ── Load checkpoint (11D obs) ─────────────────────────────────────────────
    # Load WITHOUT setting env yet — the obs space doesn't match (11D vs 12D).
    # We extend the obs dim BEFORE attaching the new env.
    checkpoint_path = str(project_root / "rl" / "ppo_curriculum_v2.zip")
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device=device)

    # ── Extend observation dimension: 11D → 12D ───────────────────────────────
    # Adds a new zero-weight input column for tyre_life to both actor and critic.
    # The policy behaves identically to v2 on step 0 (zero weight on new dim).
    # Over training, the network learns to read obs[11] = tyre_life.
    print("[Train] Extending obs dim: 11D → 12D (zero-padding new tyre_life input)")
    extend_obs_dim(model, old_dim=11, new_dim=12)

    # ── Attach the new 12D env ────────────────────────────────────────────────
    model.set_env(env)

    # ── Learning rate: same as d15 continuation ───────────────────────────────
    # 1e-4 → 1e-6 cosine.  We can afford this (higher than d16's 5e-5) because:
    #   - We're not changing episode termination (no d16-style crash).
    #   - The obs extension starts at zero weight → no sudden gradient spike.
    #   - We WANT the network to learn the new tyre_life dimension quickly.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_tyre", ["stdout", "tensorboard"]))

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[Train] Tyre degradation training for {ADDITIONAL_STEPS:,} steps")
    print(f"        Obs: 12D (added tyre_life at index 11)")
    print(f"        Tyre wear: base=0.0003/step + 0.002×slip_angles/step")
    print(f"        Normal driving (~0.2 rad slip): tyres worn at ~1428 steps")
    print(f"        Aggressive driving (~0.5 rad slip): worn at ~769 steps")
    print(f"        Episodes still truncate at max_steps=2000 (no multi_lap)")
    print(f"        Starting LR: 1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False,   # continue global step counter from ~3M
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_tyre.zip")
    model.save(save_path)
    print(f"\n[Train] Saved tyre model to {save_path}")
    print(f"        Run evaluate.py to compare against ppo_curriculum_v2.")


if __name__ == "__main__":
    train()
