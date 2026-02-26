"""
PPO Pit Strategy v2 (Week 5 / d19).

WHAT IS THIS?
=============
d18 trained an agent to drive and pit-stop, but the agent never discovered
pitting.  This script fixes all three root causes identified in d18's failure
analysis and retrains from scratch.

WHY d18 FAILED — ROOT CAUSE SUMMARY
=====================================
d18 post-mortem revealed three independent problems, each of which alone
would prevent the agent from discovering pitting:

  1. BC class imbalance (3,400:1 no-pit to pit):
     MSE loss drove pit_signal → -1.0.  The policy started maximally
     "anti-pit" before PPO even began.

  2. gamma=0.99 discounts pit payoff to near zero:
     0.99^1000 ≈ 4e-5.  A pit at step 1000 earning +500 reward over the
     remaining 1000 steps was worth 500 × 4e-5 = 0.02 to the value function.
     That's indistinguishable from noise.

  3. Value function never saw a pit:
     Even with gamma=0.9999, if the agent never pits, Q(s,pit) stays at its
     random initialisation.  No gradient can pull pit_signal upward because
     the value function has never observed "pit → tyre reset → more reward."

D19 FIXES
=========
  Fix 1 — Balanced BC dataset (generate_dataset_pit_v2):
    Only keep episodes where the expert actually pits (pit_count >= 1).
    Pit-positive fraction rises from ~0.03% (d18) to ~5% (d19).
    BC policy now initialises with pit_signal > 0 when tyre_life < 0.3.

  Fix 2 — gamma=0.9999:
    0.9999^1000 ≈ 0.905.  The value function now sees 90% of the pit payoff
    from 1000 steps away.  The gradient signal is visible at the policy.

  Fix 3 — Forced pit exploration Stage 0 (STAGES_PIT_V2):
    Stage 0 (100k steps): env forces a pit every 500 steps.
    The value function learns Q(s,pit) >> Q(s,no-pit) from forced experience
    BEFORE the agent is required to signal pits itself.
    Stages 1–3: forced pits OFF.  Agent must signal pits.  Now it has a
    learned reason to (the value function already knows pitting is good).

TRAINING PIPELINE
=================
  Step 1: generate_dataset_pit_v2() — balanced BC data (50 pit-episodes only)
  Step 2: train_bc() — 12D → 3D BC policy
  Step 3: Build PPO with gamma=0.9999
  Step 4: Transfer BC weights into PPO actor
  Step 5: Run curriculum (STAGES_PIT_V2: 4 stages with Stage 0 forced pits)
  Step 6: Save to rl/ppo_pit_v2.zip

SAVES TO: rl/ppo_pit_v2.zip
LOGS TO:  runs/ppo_pit_v2/
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

from expert.collect_data import generate_dataset_pit_v2
from bc.train_bc import train_bc
from rl.bc_init_policy import load_bc_weights_into_ppo, verify_transfer
from rl.curriculum import CurriculumCallback, STAGES_PIT_V2
from rl.make_env import make_env_pit
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_TIMESTEPS = 1_000_000

    # ── STEP 1: Collect BALANCED expert demonstrations ────────────────────────
    # generate_dataset_pit_v2() only keeps episodes where pit_count >= 1.
    # This ensures every episode in the dataset contains at least one pit event,
    # raising the pit-positive fraction from 0.03% (d18) to ~5% (d19).
    pit_data_path = str(project_root / "bc" / "expert_data_pit_v2.npz")
    if not Path(pit_data_path).exists():
        print(f"\n[Train] Collecting BALANCED pit-stop expert data (d19 Fix 1)...")
        print(f"        Only keeping episodes with pit_count >= 1.")
        generate_dataset_pit_v2(
            num_pit_episodes=50,
            max_steps=2000,
            output_path="bc/expert_data_pit_v2.npz",
        )
    else:
        print(f"[Train] Balanced expert data already exists: {pit_data_path}")

    # ── STEP 2: Train BC policy on balanced pit demonstrations ───────────────
    # BCPolicy(state_dim=12, action_dim=3) auto-detected from the .npz file.
    # With balanced data, BC now receives meaningful pit-positive gradient signal.
    # The resulting policy should initialise pit_signal > 0 when tyre_life < 0.3.
    bc_pit_v2_path = str(project_root / "bc" / "bc_policy_pit_v2.pt")
    if not Path(bc_pit_v2_path).exists():
        print(f"\n[Train] Training BC pit policy v2 on balanced dataset...")
        train_bc(
            npz_path=pit_data_path,
            num_epochs=20,
            batch_size=256,
            learning_rate=1e-3,
            device=device,
            save_path=bc_pit_v2_path,
        )
    else:
        print(f"[Train] BC pit policy v2 already exists: {bc_pit_v2_path}")

    # ── STEP 3: Build pit-stop environment ────────────────────────────────────
    # Same env as d18 (pit_stops=True, tyre_degradation=True).
    # forced_pit_interval will be set by CurriculumCallback in Stage 0.
    # The base env starts with forced_pit_interval=0; Stage 0 sets it to 500.
    env = DummyVecEnv([make_env_pit])

    # ── STEP 4: Build PPO model with gamma=0.9999 (d19 Fix 2) ─────────────────
    # KEY CHANGE: gamma=0.99 → gamma=0.9999.
    #   d18:  0.99^1000 ≈ 4e-5  (pit payoff at step 1000 ≈ worthless)
    #   d19:  0.9999^1000 ≈ 0.905  (pit payoff at step 1000 ≈ 90% discounted)
    # The value function can now see and credit the tyre benefit from pitting.
    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),

        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        # d19 Fix 2: longer discount horizon.
        # At gamma=0.9999, rewards 1000 steps away are discounted to 90.5%
        # of their nominal value — compared to 0.004% with gamma=0.99.
        # The value function gradient now has a clear signal: pit is good.
        gamma=0.9999,

        gae_lambda=0.95,

        # Higher entropy coefficient (0.005 → 0.01) to keep pit_signal
        # exploration alive longer.  Entropy bonus prevents premature
        # collapse of pit_signal distribution to -1.0.
        clip_range=0.1,
        ent_coef=0.01,

        vf_coef=0.5,
        max_grad_norm=0.5,

        # Architecture: matches BC policy exactly for weight transfer.
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),

        verbose=1,
        device=device,
    )

    # ── STEP 5: Transfer BC weights into PPO actor ────────────────────────────
    # Same mechanism as d13 and d18.  With balanced data, the BC policy has
    # learned pit_signal > 0 when tyre_life < 0.3 — a better warm start.
    print(f"\n[Train] Transferring BC pit v2 weights into PPO actor...")
    load_bc_weights_into_ppo(model, bc_pit_v2_path, device)
    verify_transfer(model, bc_pit_v2_path, device)

    # ── STEP 6: Configure TensorBoard ────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v2", ["stdout", "tensorboard"]))

    # ── STEP 7: Build d19 curriculum (STAGES_PIT_V2 with Stage 0) ─────────────
    # STAGES_PIT_V2 has 4 stages:
    #   Stage 0: forced_pit_interval=500, grad_window=50 → ~100k steps forced pits
    #   Stage 1: forced_pit_interval=0, grad_lap_rate=0.5, low speed
    #   Stage 2: forced_pit_interval=0, grad_lap_rate=0.3, mid speed
    #   Stage 3: forced_pit_interval=0, never graduates, full speed
    callback = CurriculumCallback(stages=STAGES_PIT_V2, verbose=1)

    # ── STEP 8: Train ─────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v2 training for {TOTAL_TIMESTEPS:,} steps")
    print(f"        d19 Fixes:")
    print(f"          Fix 1 - Balanced BC dataset (pit-positive fraction ~5%)")
    print(f"          Fix 2 - gamma=0.9999 (was 0.99 in d18)")
    print(f"          Fix 3 - Stage 0 forced pits every 500 steps (~100k steps)")
    print(f"        Obs: 12D  |  Action: 3D [throttle, steer, pit_signal]")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # ── STEP 9: Save ──────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v2.zip")
    model.save(save_path)
    print(f"\n[Train] Saved pit strategy v2 to {save_path}")
    print(f"        Final curriculum stage: {callback.current_stage.name}")
    print(f"        Run evaluate.py to compare against ppo_pit (d18).")


if __name__ == "__main__":
    train()
