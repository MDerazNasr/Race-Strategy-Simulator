"""
PPO Pit Strategy v4 D26 — Three-Layer Policy Freeze (Week 5).

WHY D26?
=========
D25 froze mlp_extractor.policy_net (driving features, 18,176 params) but kept
action_net fully trainable (all 3 output rows: throttle, steer, pit).

Result: features were PERFECTLY frozen (0.000 drift), but action_net rows 0 and 1
(throttle/steer) slowly drifted over 2M steps:
  - Stage 0: good driving (ep_len=1300, rolling_rate=83%) — action_net intact
  - Stage 2: steer collapsed to 1.0 with std=0.124 — ep_len=163, crashes
  - Evaluation: reward=1275, laps=5, 0 pits (worse than d21's 1877, 7 laps, 1 pit)

Root cause: The gradient signal from value function updates and forced pit
experiences bleeds into throttle/steer rows of action_net through the Adam
optimizer. Over 2M steps, this slow leak corrupts throttle/steer even when
the features are frozen.

THE D26 FIX — THREE-LAYER POLICY FREEZE:
=========================================

All driving parameters are frozen at d21's values:

Layer 1 — mlp_extractor.policy_net (features):
  requires_grad=False on all 18,176 parameters.
  The hidden representations encoding track state are LOCKED.
  Same approach as d25 (achieved 0.000 drift in d25).

Layer 2 — action_net throttle and steer rows (rows 0, 1):
  Gradient hooks zero out the gradient for rows 0 and 1 of action_net.weight
  and elements 0, 1 of action_net.bias.
  After each backward pass, gradient[0,:] = 0, gradient[1,:] = 0.
  Adam optimizer sees zero gradient → does not update those parameters.
  The throttle and steer output mapping is LOCKED at d21's values.

Layer 3 — log_std throttle and steer dimensions (dims 0, 1):
  Gradient hook zeros out log_std gradient for dims 0 and 1.
  Exploration std for throttle and steer is LOCKED at d21's values.

ONLY TRAINABLE POLICY PARAMETERS:
  action_net.weight[2,:]  — 128 params (pit signal weight row)
  action_net.bias[2]      — 1 param   (pit signal bias)
  log_std[2]              — 1 param   (pit signal exploration std)
  Total:                    130 params

VALUE FUNCTION REMAINS FULLY TRAINABLE:
  mlp_extractor.value_net — NOT frozen (value function can adapt)
  value_net               — NOT frozen (value head can adapt)

WHY THIS IS CORRECT:
  1. The value function MUST adapt. When we add safety-net forced pits, the
     value landscape changes slightly (more pit experiences → value of worn
     tyre states improves). If value function is frozen, it cannot learn.
     Value function recalibration is acceptable — the POLICY cannot regress.

  2. The policy (all driving dims) is COMPLETELY FROZEN. The driving behavior
     at evaluation time is GUARANTEED to be identical to d21. The steer=1.0
     hard-right collapse from d25 CANNOT happen when steer row is frozen.

  3. Only the pit row can change. D21 already pits at tyre_life≈0.35. The
     safety-net experiences (forced pits at 0.25/0.15/0.08) reinforce this
     behavior without changing it. The pit row gradient can refine timing.

ENVIRONMENT CHOICE:
  Standard env (make_env_pit, NO pit_timing_reward) — same as d21's training env.

  Rationale: with frozen driving policy, we COULD use pit_timing_reward=True
  (value crisis can't destroy driving). But the timing gradient only helps IF
  the agent can voluntarily pit at different tyre_life values. Since we're
  only training 130 parameters (pit row), the gradient signal from
  pit_timing_reward might be too weak to shift timing meaningfully.

  The standard env is simpler: no crisis, and the safety net alone will keep
  pit experiences in the distribution, maintaining d21's pit behavior exactly.

EXPECTED OUTCOME:
  - Driving: IDENTICAL to d21 (all driving params frozen at d21's values)
  - Pitting: MAINTAINED (safety net keeps pit experiences in distribution)
  - Timing: SAME as d21 (pit at tyre_life≈0.35) — no timing gradient to change it
  - Reward: ≈ d21's 1877 (since driving and pitting are preserved)
  - Key success metric: reward ≥ 1877 AND pit_count ≥ 1

COMPARISON TO PREVIOUS APPROACHES:
  d22: Full fine-tuning, no forced pits → forgot pitting (all params trainable)
  d23: Frozen pit row only → mlp_extractor drifted → forgot pitting
  d24: Safety net + pit_timing_reward → value crisis destroyed driving
  d25: Frozen features + standard env → features OK but throttle/steer drifted
  d26: Frozen features + frozen throttle/steer + safety net = should work

STARTING POINT: ppo_pit_v4.zip (d21) — pit behavior intact (1 pit, 1877 reward)
SAVES TO:       rl/ppo_pit_v4_d26.zip
LOGS TO:        runs/ppo_pit_v4_d26/
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

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")   # d21 — the only one that ever pitted!

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21, pit_count=1, reward=1877)")

    # ── Build environment ──────────────────────────────────────────────────────
    # make_env_pit(): F1Env(tyre_degradation=True, pit_stops=True)
    # SAME env as d21's training → no value recalibration crisis.
    env = DummyVecEnv([make_env_pit])

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Pre-training diagnostics (d21's intact pit behavior)
    pit_w_before = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_before = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_before = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_std_before = model.policy.log_std[2].item()
    thr_std_before = model.policy.log_std[0].item()
    str_std_before = model.policy.log_std[1].item()
    print(f"\n[Diag] Pre-training weights (all should match d21):")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_before:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_before:.6f}  (steer)")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_before:.6f}  (pit)")
    print(f"       log_std[0] = {thr_std_before:.6f} (throttle std)")
    print(f"       log_std[1] = {str_std_before:.6f} (steer std)")
    print(f"       log_std[2] = {pit_std_before:.6f} (pit std)")

    # ── LAYER 1: Freeze mlp_extractor.policy_net (driving features) ────────────
    # All 18,176 hidden-layer parameters are frozen.
    # The driving representations are LOCKED at d21's values.
    # requires_grad=False means no gradient is computed for these parameters.
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Freeze action_net throttle and steer rows via gradient hooks ──
    # Gradient hooks run after each backward pass and zero out specific rows.
    # action_net.weight shape: (3, 128) — rows: [throttle=0, steer=1, pit=2]
    # action_net.bias shape:   (3,)     — elems: [throttle=0, steer=1, pit=2]
    #
    # We zero rows 0 (throttle) and 1 (steer), leaving row 2 (pit) free.
    # Adam optimizer receives zero gradient → does not update those parameters.
    # The weight tensors for throttle/steer remain IDENTICAL to d21's values.

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        """Zero gradient for throttle row (0) and steer row (1), keep pit row (2)."""
        g = grad.clone()
        g[0, :] = 0.0   # throttle row: no update
        g[1, :] = 0.0   # steer row:    no update
        # g[2, :] unchanged — pit row updates freely
        return g

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        """Zero gradient for throttle (0) and steer (1) bias, keep pit (2)."""
        g = grad.clone()
        g[0] = 0.0   # throttle bias: no update
        g[1] = 0.0   # steer bias:    no update
        # g[2] unchanged — pit bias updates freely
        return g

    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    print(f"[Freeze] Layer 2: action_net.weight rows [0,1] + bias[0,1] (gradient hooks)")

    # ── LAYER 3: Freeze log_std throttle and steer dims via gradient hook ──────
    # log_std shape: (3,) — dims: [throttle=0, steer=1, pit=2]
    # Zero out dims 0 (throttle std) and 1 (steer std), leave dim 2 (pit std) free.

    def _hook_log_std(grad: torch.Tensor) -> torch.Tensor:
        """Zero log_std gradient for throttle (0) and steer (1), keep pit (2)."""
        g = grad.clone()
        g[0] = 0.0   # throttle std: frozen
        g[1] = 0.0   # steer std:    frozen
        # g[2] unchanged — pit std updates freely
        return g

    model.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layer 3: log_std[0,1] (gradient hook)")

    # Print summary
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Freeze] Summary:")
    print(f"         Frozen (features):    {frozen_feat:,} params (requires_grad=False)")
    print(f"         Frozen (hook):        rows 0,1 of action_net + log_std[0,1]")
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 130 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── LR schedule ───────────────────────────────────────────────────────────
    # 1e-4 → 1e-6 cosine (higher than d25's 5e-5 since only 130 policy params train)
    # With so few trainable policy params, higher LR is fine for the value layers.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure curriculum callback ─────────────────────────────────────────
    # STAGES_PIT_V5: safety net at 0.25/0.15/0.08 (never 0.0).
    # D21 pits voluntarily at tyre_life≈0.35 (before any safety net fires).
    # Safety net provides backup only if voluntary pitting regresses.
    callback = CurriculumCallback(stages=STAGES_PIT_V5, verbose=1)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d26", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D26: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:       ppo_pit_v4.zip (d21, pit_count=1, reward=1877)")
    print(f"        Starting LR:         1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        Environment:         make_env_pit (standard, NO pit_timing_reward)")
    print(f"        Frozen (Layer 1):    mlp_extractor.policy_net ({frozen_feat:,} params)")
    print(f"        Frozen (Layer 2):    action_net rows 0,1 (throttle/steer) via hooks")
    print(f"        Frozen (Layer 3):    log_std[0,1] (throttle/steer stds) via hooks")
    print(f"        Trainable policy:    action_net.weight[2,:] + bias[2] + log_std[2] (130 params)")
    print(f"        Curriculum:          STAGES_PIT_V5 (safety net: 0.25→0.15→0.08)")
    print(f"        Goal:                reward≥1877 AND pit_count≥1 (preserve d21)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=False,   # step counter continues from d21's ~1M
        callback=callback,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pit_w_after = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_after = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after = model.policy.action_net.weight[1, :].abs().mean().item()
    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_after:.6f}  (was {thr_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_after:.6f}  (was {str_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_after:.6f}  (was {pit_w_before:.6f}) [PIT: can change]")

    # Check that frozen params didn't change
    state_d21 = PPO.load(checkpoint_path, device=device)

    feature_drift = max(
        (p_new - p_old).abs().max().item()
        for p_new, p_old in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            state_d21.policy.mlp_extractor.policy_net.parameters()
        )
    )
    thr_drift = (
        model.policy.action_net.weight[0, :] - state_d21.policy.action_net.weight[0, :]
    ).abs().max().item()
    str_drift = (
        model.policy.action_net.weight[1, :] - state_d21.policy.action_net.weight[1, :]
    ).abs().max().item()

    print(f"\n[Diag] Freeze verification (all should be ≈ 0.0):")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features]")
    print(f"       action_net.weight[0,:] drift    = {thr_drift:.2e}  [throttle row]")
    print(f"       action_net.weight[1,:] drift    = {str_drift:.2e}  [steer row]")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d26.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d26 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (1877, 1 pit).")
    print(f"        If driving preserved (reward≥1700) and pit_count≥1: SUCCESS.")


if __name__ == "__main__":
    train()
