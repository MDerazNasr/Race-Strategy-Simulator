"""
PPO Multi-Lap Training: Continue from ppo_curriculum_v2.zip with no episode
truncation — episodes only end on crash.

WHAT CHANGED vs PREVIOUS TRAINING RUNS
========================================
All previous runs (d13 curriculum, d15 continued) used F1Env with
max_steps=2000. Episodes truncated after 2000 steps regardless of whether
the car was still driving well. This created a hard ceiling on what the
agent could learn: it never experienced "what happens on lap 8 or 12."

Training reward plateaued at ~2,250 ep_rew_mean at the end of d15.
That plateau was the max_steps cap, not the policy's capability.

In multi-lap mode (F1Env(multi_lap=True)):
  - truncated is NEVER set
  - Episodes only end when the car goes off-track (terminated=True)
  - A perfect episode could run for 10,000+ steps (many laps)
  - The lap bonus (+100 per lap) accumulates — the agent has a strong
    gradient to keep surviving lap after lap

WHAT THE AGENT MUST LEARN IN MULTI-LAP MODE
=============================================
Standard mode taught: "go fast, stay on track for 200 seconds."
Multi-lap mode teaches: "go fast, stay on track INDEFINITELY."

The difference is subtle but important:
  - The agent must learn to be consistent, not just occasionally fast
  - It must manage the DynamicCar's tyre dynamics across many laps,
    not just one
  - The value function must estimate multi-lap returns, not just
    the next 2000 steps

WHY CONTINUE FROM V2 INSTEAD OF RETRAINING
============================================
ppo_curriculum_v2.zip already knows:
  - How to follow the track at up to 27 m/s
  - How to manage tyre slip for sustained periods
  - The lap bonus reward structure (trained with it in d15)

We just need it to adapt to the new episode termination condition.
The first few rollouts will show longer episodes as the policy discovers
it's not being cut off — the value function will gradually update
to reflect multi-lap returns.

EXPECTED TRAINING PROGRESSION
===============================
  Early (0-200k):   Episodes still short — policy hasn't adapted yet,
                    crashes frequently as it pushes into uncharted territory.
  Mid (200k-800k):  Episodes getting longer, ep_rew_mean climbing past
                    previous plateau of ~2,250.
  Late (800k-2M):   Sustained multi-lap episodes, reward > 5,000+
                    if the agent completes 5+ laps per episode.

Saves to: rl/ppo_multi_lap.zip
Logs to:  runs/ppo_multi_lap/
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

from rl.make_env import make_env_multi_lap
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    ADDITIONAL_STEPS = 2_000_000

    # ── Multi-lap environment ────────────────────────────────────────────────
    # Episodes only end on crash. No 2000-step truncation.
    env = DummyVecEnv([make_env_multi_lap])

    # ── Load checkpoint ──────────────────────────────────────────────────────
    # Start from the best model: ppo_curriculum_v2 (3M steps, 27 m/s, 17 laps).
    checkpoint_path = str(project_root / "rl" / "ppo_curriculum_v2.zip")
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)

    # ── Learning rate: even lower for fine-tuning ────────────────────────────
    # v1 → v2 used 1e-4 → 1e-6. v2 → multi_lap uses 5e-5 → 1e-6.
    # The policy is near-optimal for single-lap driving; we only want it to
    # adapt its value estimates and corner-management to longer episodes.
    # Smaller LR = smaller policy perturbation during adaptation.
    model.learning_rate = cosine_schedule(initial_lr=5e-5, min_lr=1e-6)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_multi_lap", ["stdout", "tensorboard"]))

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\n[Train] Multi-lap training for {ADDITIONAL_STEPS:,} steps")
    print(f"        Episodes end only on crash (no 2000-step truncation)")
    print(f"        Starting LR: 5e-5 (cosine decay to 1e-6 over 2M steps)")
    print(f"        Lap bonus: +100 per lap crossing (from d14)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False,   # continue global step counter from ~3M
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_multi_lap.zip")
    model.save(save_path)
    print(f"\n[Train] Saved multi-lap model to {save_path}")
    print(f"        Run evaluate.py to compare against ppo_curriculum_v2.")


if __name__ == "__main__":
    train()
