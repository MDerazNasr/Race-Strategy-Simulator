"""
PPO Training with Curriculum Learning on DynamicCar.

HOW THIS DIFFERS FROM train_ppo_bc_init.py
===========================================
train_ppo_bc_init.py ran 300k steps with fixed physics and reward weights.
This script adds a CurriculumCallback that:
  1. Starts with a speed cap (~8 m/s) and high stability penalties
  2. Automatically advances through 3 stages as the agent improves
  3. Ends with full-speed racing and standard reward weights

The PPO hyperparameters and BC initialisation are unchanged.
Only two things are different:
  - total_timesteps: 300k -> 1_000_000
    (curriculum needs time to progress through all 3 stages)
  - callback: RacingMetricsCallback -> CurriculumCallback
    (CurriculumCallback also logs racing metrics internally)

WHY KEEP EVERYTHING ELSE THE SAME?
  Curriculum learning is ORTHOGONAL to the other stability improvements.
  ent_coef, clip_range, cosine LR, BC init -- all still beneficial.
  Adding curriculum on top of those improvements maximises stability.
  Changing too many things at once makes it impossible to know what worked.

EXPECTED TRAINING PROGRESSION:
  0 -- ~200k steps:    Stage 1, speed <= 8 m/s, agent learns to survive
  ~200k -- ~600k steps: Stage 2, speed <= 15 m/s, agent learns speed control
  ~600k -- 1M steps:   Stage 3, full speed, agent races

  (Exact graduation steps depend on how fast the agent learns each stage.)

TensorBoard:
  tensorboard --logdir runs/
  New panels: curriculum/stage, curriculum/lap_rate, curriculum/rolling_rate
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

from rl.bc_init_policy import load_bc_weights_into_ppo, verify_transfer
from rl.curriculum import CurriculumCallback, STAGES
from rl.make_env import make_env
from rl.schedules import cosine_schedule


def train():
    """
    Full curriculum PPO pipeline:
      1. Build env + model (same as stable training)
      2. Load BC weights (same warm start)
      3. Attach CurriculumCallback
      4. Train for 1M steps (curriculum drives stage progression)
      5. Save to rl/ppo_curriculum.zip
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # ── Total training budget ────────────────────────────────────────────────
    # 1M steps gives each stage roughly 200-400k steps to converge.
    # At ~7000 steps/sec on CPU, 1M steps takes ~2.5 minutes.
    TOTAL_TIMESTEPS = 1_000_000

    # ── Build vectorized environment ─────────────────────────────────────────
    # Identical to stable training. CurriculumCallback will modify the
    # inner env's physics and reward in-place at each stage transition.
    env = DummyVecEnv([make_env])

    # ── Build PPO model ──────────────────────────────────────────────────────
    # All hyperparameters identical to ppo_bc_stable.
    # Curriculum is applied by the callback AFTER model creation.
    model = PPO(
        policy="MlpPolicy",
        env=env,

        # Cosine LR decay: same as stable training
        # Over 1M steps the decay is slower -- good, we need bold updates
        # in Stage 1 where the policy changes a lot.
        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),

        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,

        # Entropy and clip: same conservative settings as stable training.
        # With curriculum, we start in a simple regime so ent_coef=0.005
        # keeps exploration alive without fighting a hard task.
        clip_range=0.1,
        ent_coef=0.005,

        vf_coef=0.5,
        max_grad_norm=0.5,

        # Architecture: must match BCPolicy for weight transfer
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),

        verbose=1,
        device=device,
    )

    # ── Configure TensorBoard ────────────────────────────────────────────────
    # Separate run directory so we can compare curriculum vs stable side-by-side.
    model.set_logger(configure("runs/ppo_curriculum", ["stdout", "tensorboard"]))

    # ── Transfer BC weights into PPO actor ───────────────────────────────────
    # Same warm start as before. BC policy was trained on 11D obs.
    # The curriculum starts with constrained physics, not a re-trained BC --
    # the BC prior is still valid, we just throttle the engine.
    bc_path = str(project_root / "bc" / "bc_policy_final.pt")
    load_bc_weights_into_ppo(model, bc_path, device)
    verify_transfer(model, bc_path, device)

    # ── Build curriculum callback ────────────────────────────────────────────
    # CurriculumCallback:
    #   - applies Stage 1 on training start
    #   - monitors lap completion rate every rollout
    #   - advances stages when graduation criteria are met
    #   - logs curriculum/stage, curriculum/lap_rate to TensorBoard
    callback = CurriculumCallback(stages=STAGES, verbose=1)

    # ── Run PPO training ─────────────────────────────────────────────────────
    print(f"\n[Train] Starting curriculum PPO -- {TOTAL_TIMESTEPS:,} steps")
    print(f"        Stage 1: stability (<=8 m/s)   -> grad at 50% lap rate")
    print(f"        Stage 2: speed    (<=15 m/s)   -> grad at 30% lap rate")
    print(f"        Stage 3: racing   (full speed)  -> runs to end")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # ── Save trained model ───────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_curriculum.zip")
    model.save(save_path)
    print(f"\n[Train] Saved curriculum PPO to {save_path}")
    print(f"        Final stage reached: {callback.current_stage.name}")


if __name__ == "__main__":
    train()
