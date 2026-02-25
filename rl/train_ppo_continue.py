"""
PPO Continued Training: Resume from ppo_curriculum.zip for 2M more steps.

WHY CONTINUE INSTEAD OF RETRAIN
=================================
ppo_curriculum.zip already completed:
  Stage 1 (~22k steps):   learned to survive at <=8 m/s
  Stage 2 (~82k steps):   learned to go faster at <=15 m/s
  Stage 3 (~900k steps):  learned to race at full speed (~15-25 m/s)

Restarting from scratch would waste all of Stage 1+2 (easily 100k steps
of solved curriculum) and ~900k more steps before reaching the same
policy quality. Continuing from the checkpoint means we spend ALL 2M
new steps on the hard problem: learning tyre management at 25 m/s.

WHY THE NEW REWARD DOESN'T BREAK CONTINUITY
============================================
After d14, F1Env.step() adds +100 to the reward whenever the car crosses
the start/finish line (lap_bonus). The trained policy's BEHAVIOUR hasn't
changed — only the reward signal it receives during the continued training.

PPO is on-policy: every rollout uses the CURRENT policy to generate
fresh transitions. The old rollouts (from ppo_curriculum.zip) are discarded.
The new rollouts will include lap bonuses whenever the policy completes
a lap, giving it a strong gradient toward sustained lap completion.

EXPECTED OUTCOME
================
Fixed-start eval showed PPO goes 25 m/s but crashes before completing
sustained laps. At 2M extra steps of Stage 3, the policy should:
  - Learn to modulate speed in corners to control v_y
  - Learn to survive a full 2000-step episode at ~18-22 m/s
  - Show meaningful lap-completion rate on fixed-start eval (> 0%)

Training budget:
  At ~7,000 steps/sec on CPU, 2M steps ≈ ~5 minutes.

Saves to: rl/ppo_curriculum_v2.zip
Logs to:  runs/ppo_curriculum_v2/
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

from rl.make_env import make_env
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    ADDITIONAL_STEPS = 2_000_000

    # ── Build environment ────────────────────────────────────────────────────
    # The env now includes the lap bonus (+100 per lap completion).
    # This is the only change from the original curriculum training.
    env = DummyVecEnv([make_env])

    # ── Load checkpoint ──────────────────────────────────────────────────────
    # PPO.load() reconstructs the model (architecture + weights + optimizer).
    # We pass the new env so SB3 knows the current observation/action spaces.
    checkpoint_path = str(project_root / "rl" / "ppo_curriculum.zip")
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)

    # ── Update learning rate schedule ────────────────────────────────────────
    # The original cosine schedule decayed over 1M steps (3e-4 → 1e-6).
    # For the continued run we use a fresh cosine schedule over 2M steps,
    # starting from a lower initial LR (1e-4) since the policy is already
    # well-trained and we want fine-tuning, not large updates.
    #
    # WHY LOWER STARTING LR?
    #   A warm-started policy already has good behaviour. Large gradient
    #   steps could overwrite what was learned. Starting at 1e-4 instead
    #   of 3e-4 gives smaller but more targeted updates — fine-tuning mode.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure TensorBoard ────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_curriculum_v2", ["stdout", "tensorboard"]))

    # ── Run continued training ───────────────────────────────────────────────
    # reset_num_timesteps=False: continue the global step counter from where
    # the original run left off (~1M). This means TensorBoard shows a
    # continuous curve from 0 to 3M steps, not two separate runs.
    print(f"\n[Train] Continuing Stage 3 training for {ADDITIONAL_STEPS:,} more steps")
    print(f"        Starting LR: 1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        Reward: includes +100 lap bonus (new since d14)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False,   # continue step count from checkpoint
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_curriculum_v2.zip")
    model.save(save_path)
    print(f"\n[Train] Saved continued model to {save_path}")
    print(f"        Run evaluate.py to compare against original curriculum model.")


if __name__ == "__main__":
    train()
