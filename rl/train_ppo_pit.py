"""
PPO Pit Stop Training (Week 5 / d18).

WHAT IS THIS?
=============
This script trains a PPO agent that can BOTH drive the car AND decide when
to pit stop — the first true race strategy decision in this project.

The pit stop mechanic:
  - Action space: 3D [throttle, steer, pit_signal]
    pit_signal ∈ [-1, 1]; fires a pit when > 0 and cooldown is zero.
  - Pitting resets tyre_life to 1.0 (fresh grip) but costs -200 reward.
  - pit_cooldown = 100 steps after each pit (prevents instant re-pitting).
  - Observation: 12D (same as d17 tyre env — tyre_life at index 11).

WHY TRAIN FROM SCRATCH INSTEAD OF CONTINUING FROM ppo_tyre?
============================================================
ppo_tyre was trained with a 2D action space [throttle, steer].
Adding the 3rd dimension (pit_signal) changes the policy's output layer:
  Old: action_net = Linear(128, 2)   ← from ppo_tyre
  New: action_net = Linear(128, 3)   ← for pit policy

There is no weight transfer trick for NEW action output units.
The throttle/steer outputs could theoretically be copied, but the pit_signal
output starts at zero — giving a policy that NEVER pits, which is a bad
initialization for learning pit strategy.

Instead we use the proven warm-start pipeline:
  1. Collect expert demonstrations (12D obs, 3D actions).
     Expert pits when tyre_life < 0.3 (30% grip).
  2. Train a BC policy (12D → 3D) via supervised learning.
     BC policy learns: track-following + when to pit.
  3. Transfer BC weights into PPO actor (same as d13 curriculum pipeline).
  4. Run curriculum + PPO for 1M steps.

This mirrors exactly how ppo_curriculum_v2 was built — just with a 3D
action space and a pit-aware expert/BC.

THE OPTIMAL PIT STRATEGY (what the agent must discover)
========================================================
Episode length: 2000 steps.
Tyre wear at normal driving (~0.2 rad slip): 0.0007/step.
  → tyre_life = 1.0 → 0.3 in ~1000 steps (half the episode).

At tyre_life=0.3, grip = 0.3 * 1.5 = 0.45 (30% of fresh).
At tyre_life=0.1 (floor), grip = 0.1 * 1.5 = 0.15 (10% of fresh).

Fresh tyres (tyre_life=1.0) earn significantly more progress reward
per step than worn tyres.  A rough estimate:
  - Fresh: progress ≈ 0.9/step
  - Worn (0.1): progress ≈ 0.4/step (speed reduced ~50%)
  - Difference: +0.5/step
  - Steps remaining after pit at step 1000: ~1000
  - Gain from pit: 0.5 × 1000 = 500 >> 200 (pit cost)

The agent SHOULD learn to pit once at ~step 1000 when tyre_life ≈ 0.3.
  - Drive step 0-1000: fresh tyres, high pace.
  - Pit at step ~1000: pay 200, get fresh tyres.
  - Drive step 1001-2000: fresh tyres again, high pace.

This doubles tyre-limited lap performance compared to no pit.
The BC expert shows exactly this behavior — the PPO refines the exact
timing based on the reward gradient.

CURRICULUM STILL APPLIES
=========================
The CurriculumCallback modifies car.max_accel and reward_fn in-place.
These changes are orthogonal to the pit mechanic — the 3D action space
doesn't affect how the curriculum works.  The agent learns:
  Stage 1: Stay alive at low speed while learning pit timing.
  Stage 2: Increase speed while maintaining pit strategy.
  Stage 3: Full racing speed with optimal pit decisions.

SAVES TO: rl/ppo_pit.zip
LOGS TO:  runs/ppo_pit/
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

from expert.collect_data import generate_dataset_pit
from bc.train_bc import train_bc
from rl.bc_init_policy import load_bc_weights_into_ppo, verify_transfer
from rl.curriculum import CurriculumCallback, STAGES
from rl.make_env import make_env_pit
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_TIMESTEPS = 1_000_000

    # ── STEP 1: Collect expert demonstrations (12D obs, 3D actions) ──────────
    # ExpertDriver(include_pit=True) pits when tyre_life < 0.3.
    # 50 episodes × ~2000 steps ≈ 100k raw samples.  After edge-state
    # duplication, expect ~110-130k samples in the dataset.
    pit_data_path = str(project_root / "bc" / "expert_data_pit.npz")
    if not Path(pit_data_path).exists():
        print(f"\n[Train] Collecting pit-stop expert data...")
        generate_dataset_pit(
            num_episodes=50,
            max_steps=2000,
            output_path="bc/expert_data_pit.npz",
        )
    else:
        print(f"[Train] Expert data already exists: {pit_data_path}")

    # ── STEP 2: Train BC policy on pit demonstrations ─────────────────────────
    # BCPolicy(state_dim=12, action_dim=3) is auto-detected from the .npz file.
    # Training produces a 12D → 3D policy that knows:
    #   - How to follow the track (from throttle/steer targets).
    #   - When to pit (pit_signal = +1 when tyre_life < 0.3).
    bc_pit_path = str(project_root / "bc" / "bc_policy_pit.pt")
    if not Path(bc_pit_path).exists():
        print(f"\n[Train] Training BC pit policy (12D obs → 3D actions)...")
        train_bc(
            npz_path=pit_data_path,
            num_epochs=20,
            batch_size=256,
            learning_rate=1e-3,
            device=device,
            save_path=bc_pit_path,
        )
    else:
        print(f"[Train] BC pit policy already exists: {bc_pit_path}")

    # ── STEP 3: Build pit-stop environment ────────────────────────────────────
    # F1Env(tyre_degradation=True, pit_stops=True):
    #   - 12D observation space (same as tyre env)
    #   - 3D action space [throttle, steer, pit_signal]
    env = DummyVecEnv([make_env_pit])

    # ── STEP 4: Build PPO model with 3D action space ─────────────────────────
    # PPO is created fresh (cannot continue from ppo_tyre — action space changed).
    # net_arch must match BCPolicy exactly for weight transfer.
    model = PPO(
        policy="MlpPolicy",
        env=env,

        # Cosine LR: same as ppo_curriculum pipeline (d13).
        # Starting at 3e-4 allows bold updates in Stage 1 where the policy
        # changes a lot as it learns both driving and pit strategy.
        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),

        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,

        # Conservative clip and entropy: same as ppo_curriculum (d13).
        # ent_coef=0.005 keeps exploration alive without fighting a hard task.
        clip_range=0.1,
        ent_coef=0.005,

        vf_coef=0.5,
        max_grad_norm=0.5,

        # Architecture: must match BCPolicy for weight transfer.
        # SB3 auto-creates:
        #   policy_net[0]: Linear(12, 128)  ← matches BC net[0]
        #   policy_net[2]: Linear(128, 128) ← matches BC net[2]
        #   action_net:    Linear(128, 3)   ← matches BC net[4]  (3D output)
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),

        verbose=1,
        device=device,
    )

    # ── STEP 5: Transfer BC weights into PPO actor ────────────────────────────
    # Same mechanism as d13 (load_bc_weights_into_ppo).
    # The auto-detection in bc_init_policy.py reads:
    #   state_dim  = ckpt["net.0.weight"].shape[1]   → 12
    #   action_dim = ckpt["net.4.weight"].shape[0]   → 3
    # So the transfer works unchanged for the 12D→3D pit policy.
    print(f"\n[Train] Transferring BC pit weights into PPO actor...")
    load_bc_weights_into_ppo(model, bc_pit_path, device)
    verify_transfer(model, bc_pit_path, device)

    # ── STEP 6: Configure TensorBoard ────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit", ["stdout", "tensorboard"]))

    # ── STEP 7: Build curriculum callback ────────────────────────────────────
    # Reuses the same STAGES as ppo_curriculum (d13).
    # The callback modifies car.max_accel and reward_fn in-place.
    # These changes are orthogonal to pit stops — 3D actions work at all speeds.
    callback = CurriculumCallback(stages=STAGES, verbose=1)

    # ── STEP 8: Train ─────────────────────────────────────────────────────────
    print(f"\n[Train] Pit-stop PPO training for {TOTAL_TIMESTEPS:,} steps")
    print(f"        Obs: 12D (tyre_life at index 11)")
    print(f"        Action: 3D [throttle, steer, pit_signal]")
    print(f"        Pit fires when: pit_signal > 0 AND cooldown == 0")
    print(f"        Pit effect: tyre_life → 1.0, reward -= 200, cooldown = 100 steps")
    print(f"        Optimal pit: ~step 1000 when tyre_life ≈ 0.3")
    print(f"        Curriculum: Stage 1 (8 m/s) → Stage 2 (15 m/s) → Stage 3 (full)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # ── STEP 9: Save ──────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit.zip")
    model.save(save_path)
    print(f"\n[Train] Saved pit-stop PPO to {save_path}")
    print(f"        Final curriculum stage: {callback.current_stage.name}")
    print(f"        Run evaluate.py to compare against ppo_tyre.")


if __name__ == "__main__":
    train()
