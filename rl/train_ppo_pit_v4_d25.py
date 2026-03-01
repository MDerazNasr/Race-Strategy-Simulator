"""
PPO Pit Strategy v4 D25 — Frozen Driving Features + Safety Net (Week 5).

WHY D25?
=========
D22-D24 all failed to preserve pit behavior during fine-tuning:
  d22 (2M fine-tune, standard env, no forced pits):
    pit_signal → -1.0. Better driving outweighed pit benefit. Forgotten.
  d23 (frozen pit row + pit_timing_reward):
    Output-row frozen (drift < 0.01%) but mlp_extractor changed →
    new features × frozen weights = pit_signal < 0. Behavior not frozen.
  d24 (safety-net curriculum + pit_timing_reward):
    Loading d21 into pit_timing_reward env → value recalibration crisis
    (ep_rew_mean = -1470) → steer collapsed to -0.74 hard left, std=0.144
    → deterministic policy crashes in 23 steps. Stochastic training masked it.

ROOT CAUSES (shared across d22-d24):
  1. Reward change (pit_timing_reward): changes the value landscape →
     recalibration crisis → large gradient updates → driving policy destroyed.
  2. Safety net alone: keeps pit experiences in distribution, but if driving
     is destroyed (d24), there's nothing to build on.

THE D25 FIX — TWO-PART SOLUTION:
  Fix 1 — Standard env (NO pit_timing_reward):
    D21 was trained on the standard pit env (pit_timing_reward=False).
    Loading d21 into the SAME env means NO value recalibration crisis.
    Evidence: d22 used the standard env and had NO crisis (ep_rew_mean
    never crashed — stayed positive throughout 2M steps).

  Fix 2 — Freeze mlp_extractor.policy_net entirely:
    The driving behavior lives in mlp_extractor.policy_net (hidden features).
    In SB3 with net_arch=dict(pi=[128,128], vf=[128,128]):
      mlp_extractor.policy_net:  Linear(12,128)→ReLU→Linear(128,128)→ReLU
      mlp_extractor.value_net:   Linear(12,128)→ReLU→Linear(128,128)→ReLU
      action_net:                Linear(128, 3)   ← throttle, steer, pit
      value_net:                 Linear(128, 1)

    Freezing mlp_extractor.policy_net means:
      - D21's hidden features are LOCKED (no gradient touches them)
      - The driving behavior represented in those features CANNOT regress
      - Only action_net (the 3D output layer) can change:
          action_net.weight[2,:], bias[2] → pit signal timing
          action_net.weight[0,1,:], bias[0,1] → throttle/steer (also trainable)
      - mlp_extractor.value_net and value_net are UNFROZEN:
          Value function can still adapt to curriculum threshold changes
          (minor reward shifts from different forced_pit_threshold values)

    Guarantee: deterministic policy = frozen_features × action_net_weights.
    Since frozen_features are identical to d21's features, and action_net
    starts at d21's weights, the policy at step 0 is IDENTICAL to d21.
    Subsequent updates can only change action_net — driving is protected.

CURRICULUM (STAGES_PIT_V5):
  All stages use the standard pit env (no pit_timing_reward).
  Forced pits kept at non-zero thresholds to maintain pit experiences:

  Stage 0 (~500k steps): forced_pit_threshold=0.25
    D21's voluntary pit fires at tyre_life ≈ 0.35 (before 0.25 threshold).
    Safety net never fires while voluntary pitting works.
    Pit experiences stay in the training distribution.

  Stage 1 (~1M steps): forced_pit_threshold=0.15
    Safety net only fires if voluntary pitting collapses below 0.15.
    Gap 0.15–0.35: agent must cover independently.

  Stage 2 (~500k steps): forced_pit_threshold=0.08
    Emergency safety net only. Near-undriveable tyres at 0.08.
    Any functional policy pits well before this.

WHY LOWER LR (5e-5 → 5e-7):
  Fewer parameters train (only action_net + value heads).
  The action_net is a 128→3 linear layer — very small capacity.
  Lower LR prevents overshooting the narrow action manifold.
  Standard 1e-4 is designed for full-network updates. With a frozen
  backbone, 5e-5 is more appropriate.

EXPECTED OUTCOME:
  - Driving: PRESERVED (mlp_extractor.policy_net frozen = d21 driving)
  - Pitting: MAINTAINED (safety net keeps pit experiences in distribution)
  - Timing: IMPROVED? (action_net free to shift pit_signal threshold)
  - At worst: d21 quality (1877 reward, 7 laps, 1 pit)
  - At best: d21 driving + refined pit timing (target: pit at tyre_life ≈ 0.30)

STARTING POINT: ppo_pit_v4.zip (d21) — pit behavior intact (1 pit, 1877 reward)
SAVES TO:       rl/ppo_pit_v4_d25.zip
LOGS TO:        runs/ppo_pit_v4_d25/
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

from rl.make_env import make_env_pit          # standard env — NO pit_timing_reward
from rl.curriculum import CurriculumCallback, STAGES_PIT_V5
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")   # d21 — NOT d22/d23/d24!

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21, pit_count=1, reward=1877)")

    # ── Build environment ──────────────────────────────────────────────────────
    # make_env_pit(): F1Env(tyre_degradation=True, pit_stops=True)
    # SAME env as d21's training env — NO value recalibration crisis.
    # The reward structure is IDENTICAL to what d21 was trained on.
    env = DummyVecEnv([make_env_pit])

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Diagnostic: verify d21 pit row is intact
    pit_weights_mean = model.policy.action_net.weight[2, :].abs().mean().item()
    pit_log_std      = model.policy.log_std[2].item()
    print(f"\n[Diag] Pre-training pit row (should be non-trivial for d21):")
    print(f"       action_net.weight[2,:] abs_mean = {pit_weights_mean:.6f}")
    print(f"       log_std[2]                      = {pit_log_std:.6f}")

    # ── FREEZE mlp_extractor.policy_net ───────────────────────────────────────
    # This is the core of d25. We freeze ALL parameters in the policy's shared
    # hidden layers. Only the output heads (action_net + log_std) can change.
    #
    # SB3 architecture (net_arch=dict(pi=[128,128], vf=[128,128])):
    #   mlp_extractor.policy_net  → frozen (driving features locked)
    #   mlp_extractor.value_net   → unfrozen (value function can adapt)
    #   action_net                → unfrozen (pit timing can improve)
    #   value_net                 → unfrozen (value head can adapt)
    #   log_std                   → unfrozen (exploration std can adapt)
    #
    # requires_grad=False means no gradient flows through these parameters.
    # Adam optimizer will not update them (no grad = no update).
    frozen_params = 0
    for name, param in model.policy.mlp_extractor.policy_net.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    print(f"\n[Freeze] Froze mlp_extractor.policy_net ({frozen_params:,} parameters)")

    # Count trainable parameters for verification
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"[Freeze] Trainable: {trainable:,} / {total:,} parameters")
    print(f"[Freeze] (Trainable = action_net[3×128+3] + log_std[3] + value layers)")

    # ── Lower LR for frozen-backbone fine-tuning ───────────────────────────────
    # 5e-5 → 5e-7 cosine (2× lower than d22/d23/d24's 1e-4 → 1e-6)
    # Rationale: only the 3-row action_net (387 params) trains.
    # Standard 1e-4 was designed for full-network updates.
    # With frozen backbone, smaller LR prevents overshooting the narrow
    # action manifold.
    model.learning_rate = cosine_schedule(initial_lr=5e-5, min_lr=5e-7)

    # ── Configure curriculum callback ─────────────────────────────────────────
    # STAGES_PIT_V5: safety net thresholds [0.25, 0.15, 0.08]
    # All thresholds are BELOW d21's natural voluntary pit at tyre_life=0.35,
    # so voluntary pitting fires first. Safety net only activates on regression.
    callback = CurriculumCallback(stages=STAGES_PIT_V5, verbose=1)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d25", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D25: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:    ppo_pit_v4.zip (d21, pit_count=1, reward=1877)")
    print(f"        Starting LR:      5e-5 (cosine decay to 5e-7 over 2M steps)")
    print(f"        Environment:      make_env_pit (standard, NO pit_timing_reward)")
    print(f"        Frozen:           mlp_extractor.policy_net ({frozen_params:,} params)")
    print(f"        Trainable:        action_net + log_std + value layers ({trainable:,} params)")
    print(f"        Curriculum:       STAGES_PIT_V5 (safety-net, never threshold=0.0)")
    print(f"        Stage schedule:")
    for s in STAGES_PIT_V5:
        print(f"          {s.name}")
        print(f"            forced_pit_threshold={s.forced_pit_threshold:.2f}, "
              f"grad_window={s.grad_window} rollouts (~{s.grad_window*2048//1000}k steps)")
    print(f"        Goal: preserve d21 driving (reward≥1877) AND pit_count≥1")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=False,   # step counter continues from d21's ~1M
        callback=callback,
    )

    # ── Post-training diagnostic ───────────────────────────────────────────────
    pit_weights_post = model.policy.action_net.weight[2, :].abs().mean().item()
    pit_log_std_post = model.policy.log_std[2].item()
    print(f"\n[Diag] Post-training pit row:")
    print(f"       action_net.weight[2,:] abs_mean = {pit_weights_post:.6f}  "
          f"(was {pit_weights_mean:.6f})")
    print(f"       log_std[2]                      = {pit_log_std_post:.6f}  "
          f"(was {pit_log_std:.6f})")

    # Verify frozen params didn't change
    drift = 0.0
    state_d21 = PPO.load(checkpoint_path, device=device)
    for (name, p_new), (_, p_old) in zip(
        model.policy.mlp_extractor.policy_net.named_parameters(),
        state_d21.policy.mlp_extractor.policy_net.named_parameters()
    ):
        drift += (p_new - p_old).abs().max().item()
    print(f"\n[Diag] mlp_extractor.policy_net max weight drift = {drift:.2e}")
    print(f"       (should be ≈ 0.0 — frozen params must not change)")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d25.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d25 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (1877, 1 pit).")


if __name__ == "__main__":
    train()
