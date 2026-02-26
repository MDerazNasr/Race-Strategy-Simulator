"""
PPO Pit Strategy v3 (Week 5 / d20).

WHAT IS THIS?
=============
A third attempt at teaching the agent when to pit.  d18 and d19 both
identified the right root causes but had implementation gaps that prevented
the fixes from taking effect.  d20 closes those gaps.

d19 POST-MORTEM: what was implemented vs what actually happened
================================================================
d19 implemented:
  Fix 1 — generate_dataset_pit_v2(): balanced dataset (only pit-containing episodes).
           Result: pit-positive fraction 0.03% → 0.10%. Still 0.10%.
           Still 1009:1 class imbalance within each episode.
           BC policy still initialised pit_signal ≈ -1.0.

  Fix 2 — gamma=0.9999: correct and necessary. 0.9999^1000 ≈ 0.905.
           The value function CAN now see the pit payoff — but only if the
           agent actually pits. If pit never happens, Q(s,pit) never updates.

  Fix 3 — forced_pit_interval=500 in Stage 0.
           IMPLEMENTATION GAP: during Stage 0, ep_len ≈ 46-50.
           Agent crashed before step 500. Forced pits NEVER fired.
           Stage 0 was identical to d18's Stage 1 — zero pit experiences.

Result: pit_signal = -1.000, std = 0.000. Same failure as d18.

d20 FIXES: closing the implementation gaps
==========================================
Fix A — Weighted BC loss (bc/train_bc.py):
  Class imbalance fixed at the gradient level, not at the data level.
  pit_class_weight=1000 upweights pit-positive samples 1000x in the loss.
  Effective gradient ratio: 86×1000 vs 86,752×1 ≈ 1:1 (was 1:1009).
  BC network now receives a real gradient signal from pit-positive samples.
  Expected: BC policy initialises with pit_signal > 0 when tyre_life < 0.3.

Fix B — Zero-initialize pit_signal output row (rl/bc_init_policy.py):
  Even with Fix A, the BC hidden layers learn to encode driving features in
  ways that correlate "normal driving state" with "no pit".  The pit output
  row of action_net may still start biased negative even after weight training.

  After BC weight transfer: zero only the pit_signal ROW of action_net.
  Result: pit_signal output starts as N(0, σ) → P(pit_signal > 0) = 0.5.
  Maximum initial pit exploration, zero BC prior on pit timing.
  Hidden layers (driving knowledge) are fully preserved.

Fix C — Gradual forced-pit removal (STAGES_PIT_V3, rl/curriculum.py):
  Stage 0 (interval=50): fires at step 50, 100, 150, ...
    Even in 50-step episodes, EVERY episode sees a forced pit (at step 50).
    Value function learns Q(s_50, pit) from rollout 1.
    Unlike d19's interval=500, this ALWAYS fires.

  Stage 1 (interval=100): half frequency.
    Forced pits still active at medium frequency.
    Agent supplements with own pit signals between forced pits.
    The pit_signal row now has ~100k steps of value gradient.

  Stage 2 (interval=0): agent fully responsible.
    No training wheels. Value function bootstrapped. Policy has gradient history.

  Stage 3 (interval=0): full racing pace + autonomous pit strategy.

ALSO RETAINED FROM d19:
  - gamma=0.9999 (Fix 2 from d19 — correct and kept).
  - generate_dataset_pit_v2() for balanced BC data (combined with Fix A).

SAVES TO: rl/ppo_pit_v3.zip
LOGS TO:  runs/ppo_pit_v3/
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
from rl.bc_init_policy import load_bc_weights_into_ppo, verify_transfer, zero_pit_signal_output
from rl.curriculum import CurriculumCallback, STAGES_PIT_V3
from rl.make_env import make_env_pit
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_TIMESTEPS = 1_000_000

    # ── STEP 1: Collect balanced expert demonstrations ────────────────────────
    # Reuse generate_dataset_pit_v2() from d19 (only keeps pit-containing episodes).
    # The data itself is unchanged — the weighted BC loss (Fix A) is applied at
    # training time, not at data collection time.
    pit_data_path = str(project_root / "bc" / "expert_data_pit_v2.npz")
    if not Path(pit_data_path).exists():
        print(f"\n[Train] Collecting balanced pit-stop expert data...")
        generate_dataset_pit_v2(
            num_pit_episodes=50,
            max_steps=2000,
            output_path="bc/expert_data_pit_v2.npz",
        )
    else:
        print(f"[Train] Balanced expert data already exists: {pit_data_path}")

    # ── STEP 2: Train BC policy with WEIGHTED LOSS (d20 Fix A) ───────────────
    # pit_class_weight=1000: pit-positive samples contribute 1000x more loss.
    # Effective class ratio after weighting:
    #   86 pit-positive × 1000 = 86,000  vs  86,752 pit-negative × 1 = 86,752
    #   Ratio: ~1:1  (was 1:1009 with standard MSE)
    # The BC network should learn: tyre_life < 0.3 → pit_signal → +1.0.
    bc_pit_v3_path = str(project_root / "bc" / "bc_policy_pit_v3.pt")
    if not Path(bc_pit_v3_path).exists():
        print(f"\n[Train] Training weighted BC policy (d20 Fix A: pit_class_weight=1000)...")
        train_bc(
            npz_path=pit_data_path,
            num_epochs=20,
            batch_size=256,
            learning_rate=1e-3,
            device=device,
            save_path=bc_pit_v3_path,
            pit_dim=2,               # pit_signal is action dimension 2
            pit_class_weight=1000.0, # upweight pit-positive 1000x
        )
    else:
        print(f"[Train] Weighted BC policy already exists: {bc_pit_v3_path}")

    # ── STEP 3: Build pit-stop environment ────────────────────────────────────
    # Same as d18/d19. forced_pit_interval will be set by CurriculumCallback.
    env = DummyVecEnv([make_env_pit])

    # ── STEP 4: Build PPO with gamma=0.9999 (retained from d19) ──────────────
    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),

        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        # gamma=0.9999 retained from d19 (Fix 2 — correct and necessary).
        # 0.9999^1000 ≈ 0.905: value function sees 90% of pit payoff from 1000 steps.
        gamma=0.9999,

        gae_lambda=0.95,

        # ent_coef=0.01 — same as d19.
        # Note: in d20, the MAIN entropy fix is zero_pit_signal_output()
        # (starting at P=0.5 rather than P≈0.0).  Entropy coefficient alone
        # is not enough to overcome the BC prior pushing pit toward -1.0.
        clip_range=0.1,
        ent_coef=0.01,

        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),

        verbose=1,
        device=device,
    )

    # ── STEP 5: Transfer BC weights into PPO actor ────────────────────────────
    # Step 5a: full weight transfer (hidden layers + all output rows including pit)
    print(f"\n[Train] Transferring weighted BC weights into PPO actor...")
    load_bc_weights_into_ppo(model, bc_pit_v3_path, device)

    # Step 5b: verify the transfer is exact BEFORE zeroing the pit row.
    # verify_transfer checks ALL weights including action_net.
    # If we zero first, the pit row diff would show as a false mismatch.
    verify_transfer(model, bc_pit_v3_path, device)

    # Step 5c: zero the pit_signal output row (d20 Fix B).
    # CALL ORDER IS CRITICAL: must come AFTER verify_transfer.
    # This gives pit_signal: Gaussian(mean=0, std≈0.6) → P(pit)=0.5 initially.
    # Throttle and steer outputs are unchanged from BC.
    zero_pit_signal_output(model, pit_action_dim=2)

    # ── STEP 6: Configure TensorBoard ────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v3", ["stdout", "tensorboard"]))

    # ── STEP 7: Build d20 curriculum (STAGES_PIT_V3) ──────────────────────────
    # Stage 0: forced_pit_interval=50  → fires in every episode (even ep_len=50)
    # Stage 1: forced_pit_interval=100 → half frequency, agent supplements
    # Stage 2: forced_pit_interval=0   → agent fully autonomous
    # Stage 3: forced_pit_interval=0   → full racing
    callback = CurriculumCallback(stages=STAGES_PIT_V3, verbose=1)

    # ── STEP 8: Train ─────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v3 training for {TOTAL_TIMESTEPS:,} steps")
    print(f"        d20 Fixes vs d19:")
    print(f"          Fix A — Weighted BC loss: pit_class_weight=1000 (was standard MSE)")
    print(f"          Fix B — Zero pit output row: P(pit)=0.5 at init (was ≈0.0)")
    print(f"          Fix C — forced_pit_interval=50 in Stage 0 (was 500, never fired)")
    print(f"        Retained from d19:")
    print(f"          gamma=0.9999 (0.9999^1000=0.905, pit payoff visible)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # ── STEP 9: Save ──────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v3.zip")
    model.save(save_path)
    print(f"\n[Train] Saved pit strategy v3 to {save_path}")
    print(f"        Final curriculum stage: {callback.current_stage.name}")
    print(f"        Run evaluate.py to see if pit discovery succeeded.")


if __name__ == "__main__":
    train()
