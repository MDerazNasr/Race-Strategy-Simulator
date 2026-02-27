"""
PPO Pit Strategy v4 (Week 5 / d21).

WHAT IS THIS?
=============
The fourth pit-stop training attempt.  d18-d20 all failed to discover
pitting because their forced-pit mechanisms were TIME-BASED — they fired
regardless of tyre state, sometimes pitting on fresh tyres and teaching
the agent "pitting is always bad."

d21 introduces STATE-CONDITIONAL forced pits: the environment forces a pit
ONLY when tyre_life < threshold, ensuring forced pits always happen on WORN
tyres and always produce the correct training signal.

D20 CATASTROPHIC FAILURE — ROOT CAUSE:
========================================
d20's zero-init (Fix B) gave P(pit_signal > 0) = 0.5 at every step.
Combined with interval=50, the agent pitted at step 50 (tyre_life ≈ 0.97).
Result:
  1. -200 penalty on fresh tyres in first rollout
  2. PPO immediately learns "pitting is always bad"
  3. pit_signal collapsed to -1.000 in one gradient update
  4. Driving policy destroyed (reward=-17.9, crashes at step 7)

D21 FUNDAMENTAL REDESIGN:
==========================
  OLD: forced_pit_interval=50  → pits at step 50 (tyre_life≈0.97, FRESH)
  NEW: forced_pit_threshold=0.35 → pits when tyre_life < 0.35 (WORN, ~step 929)

  The key insight: the agent must ASSOCIATE pit benefits with worn tyres.
  Time-based forcing cannot guarantee this. State-based forcing can.

D21 STARTING POINT — PPO_TYRE (not BC):
=========================================
d18-d20 all started from BC weights (12D→3D). The BC policy (even with
pit_class_weight=1000) is a weaker driver than ppo_tyre: it collapses to
short episode lengths (ep_len ≈ 30-50) where forced pits at step 929 can
never fire.

ppo_tyre (from d17, 5M+ steps) survives ~1500 steps per episode.
Starting from ppo_tyre guarantees episodes reach tyre_life=0.35 (~step 929).
load_ppo_tyre_into_ppo_pit() transfers:
  - Hidden layers + value net: from ppo_tyre (driving + value knowledge)
  - Throttle/steer rows: from ppo_tyre (action quality preserved)
  - Pit row: from bc_policy_pit_v3 (BC with pit_class_weight=1000)
  - No zero-init of pit row (d20 Fix B removed — catastrophic harm confirmed)

CURRICULUM (STAGES_PIT_V4):
============================
  Stage 0: forced_pit_threshold=0.35 (~100k steps)
    → Every episode reaches tyre_life=0.35 (~step 929).
    → Env forces pit. Value function learns: Q(s_worn, pit) >> Q(s_worn, no-pit).
    → ZERO risk of fresh-tyre -200 penalties.

  Stage 1: forced_pit_threshold=0.25 (backup for deeply worn tyres)
    → Agent must pit 0.25-0.35 on its own.
    → Env is backup only if agent fails to pit before tyre_life < 0.25.

  Stage 2: forced_pit_threshold=0.0 (agent fully autonomous, max_accel=15)
    → Value function bootstrapped with correct state→action associations.
    → Agent should now signal pit at tyre_life ≈ 0.3.

  Stage 3: full racing + autonomous pit strategy (never graduates)

ALSO RETAINED:
  - gamma=0.9999: 0.9999^1000 ≈ 0.905 (pit payoff visible to gradient)
  - Weighted BC loss for pit row: bc_policy_pit_v3 trained with 1000x weight

SAVES TO: rl/ppo_pit_v4.zip
LOGS TO:  runs/ppo_pit_v4/
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

from rl.bc_init_policy import load_ppo_tyre_into_ppo_pit
from rl.curriculum import CurriculumCallback, STAGES_PIT_V4
from rl.make_env import make_env_pit
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_TIMESTEPS = 1_000_000

    ppo_tyre_path   = str(project_root / "rl" / "ppo_tyre.zip")
    bc_pit_v3_path  = str(project_root / "bc" / "bc_policy_pit_v3.pt")

    # ── STEP 1: Verify prerequisites ──────────────────────────────────────────
    if not Path(ppo_tyre_path).exists():
        raise FileNotFoundError(
            f"ppo_tyre not found: {ppo_tyre_path}\n"
            f"Run rl/train_ppo_tyre.py (d17) first."
        )
    if not Path(bc_pit_v3_path).exists():
        raise FileNotFoundError(
            f"bc_policy_pit_v3 not found: {bc_pit_v3_path}\n"
            f"Run rl/train_ppo_pit_v3.py (d20) to generate it, or run:\n"
            f"  from bc.train_bc import train_bc\n"
            f"  train_bc(npz_path='bc/expert_data_pit_v2.npz', ..., pit_class_weight=1000)"
        )
    print(f"[Train] Prerequisites verified:")
    print(f"  ppo_tyre:         {ppo_tyre_path}")
    print(f"  bc_policy_pit_v3: {bc_pit_v3_path}")

    # ── STEP 2: Build pit-stop environment ────────────────────────────────────
    # Same factory as d18-d20. forced_pit_threshold will be set by callback.
    env = DummyVecEnv([make_env_pit])

    # ── STEP 3: Build PPO with gamma=0.9999 (retained from d19) ──────────────
    # 3D action space: [throttle, steer, pit_signal]
    # net_arch=[128,128]: must match ppo_tyre's architecture for weight transfer.
    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),

        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        # gamma=0.9999 retained from d19 (correct and necessary).
        # 0.9999^1000 ≈ 0.905: value function sees 90% of pit payoff ~1000 steps ahead.
        gamma=0.9999,

        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),

        verbose=1,
        device=device,
    )

    # ── STEP 4: Transfer ppo_tyre weights + BC pit row ────────────────────────
    # load_ppo_tyre_into_ppo_pit():
    #   - Hidden layers + value net: from ppo_tyre (1500-step survival knowledge)
    #   - Throttle/steer: from ppo_tyre action_net rows [0,1]
    #   - Pit signal: from bc_policy_pit_v3 action_net row [2] (1000x weighted BC)
    #   - NO zero-init (d20 Fix B removed — confirmed catastrophic)
    print(f"\n[Train] Transferring ppo_tyre + BC pit weights into PPO actor...")
    load_ppo_tyre_into_ppo_pit(model, ppo_tyre_path, bc_pit_v3_path, device)

    # ── STEP 5: Configure TensorBoard ─────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4", ["stdout", "tensorboard"]))

    # ── STEP 6: Build d21 curriculum (STAGES_PIT_V4) ──────────────────────────
    # Stage 0: forced_pit_threshold=0.35 → fires at ~step 929 (worn tyres only)
    # Stage 1: forced_pit_threshold=0.25 → agent pits 0.25-0.35, env backup <0.25
    # Stage 2: forced_pit_threshold=0.0  → agent fully autonomous
    # Stage 3: forced_pit_threshold=0.0  → full racing (never graduates)
    callback = CurriculumCallback(stages=STAGES_PIT_V4, verbose=1)

    # ── STEP 7: Train ──────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 training for {TOTAL_TIMESTEPS:,} steps")
    print(f"        d21 vs d20:")
    print(f"          REMOVED: zero-init pit row (d20 Fix B — catastrophic)")
    print(f"          REMOVED: time-based forced pits (interval=50/500)")
    print(f"          ADDED:   state-conditional forced pits (threshold=0.35/0.25)")
    print(f"          ADDED:   start from ppo_tyre (1500-step survival)")
    print(f"        Retained:")
    print(f"          gamma=0.9999 (pit payoff visible to value function)")
    print(f"          BC pit row with pit_class_weight=1000")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # ── STEP 8: Save ───────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4.zip")
    model.save(save_path)
    print(f"\n[Train] Saved pit strategy v4 to {save_path}")
    print(f"        Final curriculum stage: {callback.current_stage.name}")
    print(f"        Run evaluate.py to see if pit discovery succeeded.")


if __name__ == "__main__":
    train()
