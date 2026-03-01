"""
PPO Pit Strategy v4 D24 — Safety-Net Curriculum (Week 5).

WHY D24?
=========
d22 and d23 both failed to preserve pit behavior during fine-tuning:
  d22 (2M fine-tune, standard env, no forced pits):
    pit_signal → -1.0. Better driving outweighed pit benefit. Forgotten.
  d23 (frozen pit row + pit_timing_reward):
    Pit row weights frozen (drift < 0.01%), but shared mlp_extractor features
    changed → frozen_weights × new_features = pit_signal < 0. Still forgotten.

ROOT CAUSE (common to d22 and d23):
  Once forced pits are removed (threshold=0.0 in Stage 2/3 of STAGES_PIT_V4),
  pit experiences leave the training distribution. Over 2M steps, the agent
  finds a local optimum — drive well, skip pitting — and the pit association
  fades from the value function. PPO's on-policy updates reinforce the new
  no-pit equilibrium until pit_signal = -1.0000.

THE D24 FIX — PERMANENT SAFETY NET:
  Key insight: keep forced_pit_threshold > 0 at ALL stages of fine-tuning.
  Not to bootstrap (that's d21's job), but to PREVENT FORGETTING.
  A non-zero threshold guarantees at least one pit experience per episode
  if voluntary pitting fades, keeping pit experiences in the training
  distribution forever.

CURRICULUM (STAGES_PIT_V5):
  All stages use max_accel=15.0 (d21 already trains at full speed).
  All stages use pit_timing_reward=True (via make_env_pit_d23 factory).

  Stage 0 (~500k steps): forced_pit_threshold=0.25
    D21's voluntary pit fires at tyre_life ≈ 0.35 (before 0.25 threshold).
    Safety net never fires while voluntary pitting works.
    Timing bonus (tyre_life < 0.30): gradient pulls voluntary pit toward 0.29.

  Stage 1 (~1M steps): forced_pit_threshold=0.15
    Agent should now pit at ~0.29 (timing bonus zone, tyre_life < 0.30).
    Gap 0.15–0.30: ~150 steps on worn tyres agent covers independently.
    Safety net only fires if voluntary pitting collapses below 0.15.

  Stage 2 (~500k steps): forced_pit_threshold=0.08
    Emergency safety net only. At tyre_life < 0.08 the car is nearly
    undriveable (10% grip). Functional policy pits well before this.

WHY SAFETY NET THRESHOLDS BELOW D21's NATURAL PIT (0.35):
  The safety net must be BELOW the voluntary pit timing to avoid conflicts:
    threshold=0.25 → voluntary pit at 0.35 fires first (0.35 > 0.25)
    threshold=0.15 → voluntary pit at 0.29 fires first (0.29 > 0.15)
    threshold=0.08 → any voluntary pit fires first
  This means: while voluntary pitting works, forced pits never fire.
  Forced pits only activate as a CONTINGENCY for regression.

COMBINED EFFECT:
  1. pit_timing_reward=True provides positive gradient for correct timing
  2. Safety net guarantees pit experiences never leave the distribution
  3. Both together should maintain AND refine the pit behavior from d21

STARTING POINT: ppo_pit_v4.zip (d21) — pit behavior intact (1 pit per episode)
SAVES TO:       rl/ppo_pit_v4_d24.zip
LOGS TO:        runs/ppo_pit_v4_d24/
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

from rl.make_env import make_env_pit_d23
from rl.curriculum import CurriculumCallback, STAGES_PIT_V5
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")   # d21 — NOT d22/d23!

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21, pit_count=1, reward=1877)")

    # ── Build environment ──────────────────────────────────────────────────────
    # make_env_pit_d23(): F1Env(tyre_degradation=True, pit_stops=True, pit_timing_reward=True)
    #   pit_timing_reward=True stays active throughout all stages (curriculum
    #   callback does NOT touch pit_timing_reward — only forced_pit_threshold).
    #   This provides the explicit timing gradient throughout all 2M steps.
    env = DummyVecEnv([make_env_pit_d23])

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    # CRITICAL: start from d21, not d22/d23 — d21 has the intact pit behavior.
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Diagnostic: verify d21 pit row is intact
    pit_weights_mean = model.policy.action_net.weight[2, :].abs().mean().item()
    pit_log_std      = model.policy.log_std[2].item()
    print(f"\n[Diag] Pre-training pit row (should be non-trivial for d21):")
    print(f"       action_net.weight[2,:] abs_mean = {pit_weights_mean:.6f}")
    print(f"       log_std[2]                      = {pit_log_std:.6f}")

    # ── Lower LR for fine-tuning ───────────────────────────────────────────────
    # Same schedule as d22/d23: 1e-4 → 1e-6 cosine over 2M steps.
    # Fine-tuning mode — policy knows how to drive and pit; refining timing.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure curriculum callback ─────────────────────────────────────────
    # STAGES_PIT_V5:
    #   Stage 0: forced_pit_threshold=0.25, ~500k steps
    #   Stage 1: forced_pit_threshold=0.15, ~1M steps
    #   Stage 2: forced_pit_threshold=0.08, ~500k steps (never graduates)
    #
    # The callback applies Stage 0 immediately on _on_training_start().
    # It promotes stages based on step count (grad_lap_rate=0.0 → any rate passes
    # after grad_window rollouts).
    callback = CurriculumCallback(stages=STAGES_PIT_V5, verbose=1)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d24", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D24: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:    ppo_pit_v4.zip (d21, pit_count=1, reward=1877)")
    print(f"        Starting LR:      1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        Environment:      make_env_pit_d23 (pit_timing_reward=True)")
    print(f"        Curriculum:       STAGES_PIT_V5 (safety-net, never threshold=0.0)")
    print(f"        Stage schedule:")
    for s in STAGES_PIT_V5:
        print(f"          {s.name}")
        print(f"            forced_pit_threshold={s.forced_pit_threshold:.2f}, "
              f"grad_window={s.grad_window} rollouts (~{s.grad_window*2048//1000}k steps)")
    print(f"        Goal: reward > 2283 (d22) AND pit_count > 0 (unlike d22/d23)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=False,   # step counter continues from d21's ~1M
        callback=callback,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d24.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d24 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (1877, 1 pit) "
          f"and d22 (2283, 0 pits).")


if __name__ == "__main__":
    train()
