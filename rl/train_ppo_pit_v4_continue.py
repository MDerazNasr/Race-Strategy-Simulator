"""
PPO Pit Strategy v4 Continued Training (Week 5 / d22).

WHY CONTINUE FROM PPO_PIT_V4?
==============================
ppo_pit_v4 (d21) is the first pit policy to actually discover pitting:
  - Fixed-start eval: reward=1877, 7 laps, 1 pit stop (deterministic)
  - Diagnostic: pit_signal > 0 fired exactly ONCE in 1254 steps (correct timing)
  - Beats ppo_tyre (1643 reward) — pitting is provably beneficial

However, two weaknesses remain:

  Weakness 1 — Stochastic over-pitting:
    Training ep_rew_mean ≈ -840 (negative!) during Stage 3.
    Stochastic rollouts explore pit_signal > 0 too frequently.
    Each extra pit = -200 reward → noisy, negative training signal.
    More training should tighten the pit_signal distribution around
    the one correct moment, reducing stochastic over-pitting.

  Weakness 2 — Early crash (0% lap completion rate):
    Agent gets 7 laps in ~1300 steps but crashes before step 2000.
    Likely cause: post-pit tyres are fresh → agent pushes harder →
    aggressive speed on track sections → crashes.
    More training should improve post-pit driving stability.

WHAT THIS RUN DOES:
===================
  - Loads ppo_pit_v4.zip (already knows how to pit once)
  - Continues training for 2M more steps at Stage 3 (no forced pits)
  - Lower initial LR (1e-4 instead of 3e-4) — fine-tuning mode
  - Same env: make_env_pit (tyre_degradation=True, pit_stops=True)
  - No curriculum needed: ppo_pit_v4 already graduated all stages

EXPECTED OUTCOME:
=================
  - Pit timing should sharpen: agent pits closer to the optimal tyre_life
  - Stochastic ep_rew_mean should improve from -840 toward positive
  - Crash rate should decrease: better driving quality post-pit
  - Possibly: agent discovers two-stop strategy for long episodes

SAVES TO: rl/ppo_pit_v4_cont.zip
LOGS TO:  runs/ppo_pit_v4_cont/
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

from rl.make_env import make_env_pit
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    ADDITIONAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")

    # ── Build environment ──────────────────────────────────────────────────────
    # Same as d21: tyre degradation + pit stops enabled. No forced pits —
    # agent is fully autonomous (ppo_pit_v4 already graduated past them).
    env = DummyVecEnv([make_env_pit])

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded. Continuing from d21 weights.")

    # ── Lower LR for fine-tuning ───────────────────────────────────────────────
    # d21 used initial_lr=3e-4 (discovery phase).
    # d22 uses initial_lr=1e-4 (refinement phase).
    # The policy already knows HOW to pit; we want to refine WHEN and reduce
    # stochastic over-pitting — both need smaller, targeted updates.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_cont", ["stdout", "tensorboard"]))

    # ── Run continued training ─────────────────────────────────────────────────
    # reset_num_timesteps=False: step counter continues from d21's ~1M.
    # TensorBoard shows a continuous 0→3M curve.
    print(f"\n[Train] Pit Strategy v4 continued training for {ADDITIONAL_STEPS:,} more steps")
    print(f"        Starting from: ppo_pit_v4.zip (d21, reward=1877 fixed-start)")
    print(f"        Starting LR: 1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        No forced pits — agent fully autonomous")
    print(f"        Goals:")
    print(f"          - Sharpen pit timing (reduce stochastic over-pitting)")
    print(f"          - Improve post-pit driving stability (reduce crash rate)")
    print(f"          - Increase ep_rew_mean from -840 toward positive")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_cont.zip")
    model.save(save_path)
    print(f"\n[Train] Saved continued pit model to {save_path}")
    print(f"        Run evaluate.py to compare against d21.")


if __name__ == "__main__":
    train()
