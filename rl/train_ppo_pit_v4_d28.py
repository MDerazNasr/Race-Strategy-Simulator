"""
PPO Pit Strategy v4 D28 — Pit Timing Reward + Fixed LR (Week 5).

WHY D28?
=========
D27 confirmed pit_timing_reward=True is SAFE with frozen driving (no value crisis).
Training log: ep_rew_mean stayed positive throughout, approx_kl ≈ 1e-11 (policy frozen).
But d27 failed to shift pit timing: pit row barely moved (1.1% weight change over 2M steps).

ROOT CAUSE — reset_num_timesteps=False LR bug:
  d27 intended LR: 1e-4 → 1e-6 cosine
  d27 actual LR:   3.5e-5 → 1e-6 (started at 3.5e-5, not 1e-4!)

  SB3 computes: progress_remaining = 1 - (num_timesteps / total_timesteps)
  With d26's 3M steps + 2M new = 5M total:
    progress at d27 start = 1 - 3/5 = 0.40 → cosine LR ≈ 3.5e-5
  The pit row only got 3.5e-5 LR instead of 1e-4 — 3x too weak.

THE D28 FIX — reset_num_timesteps=True:
  model.learn(..., reset_num_timesteps=True)
  This makes SB3 treat d28 as a FRESH training run (progress = 1.0 at start).
  LR starts at 1e-4 (full cosine schedule from the beginning).

  Downside: TensorBoard step counter resets to 0 (not continued from 5M).
  This is FINE — we care about the model quality, not the step counter.

D28 SETUP:
  Starting point: ppo_pit_v4_d27.zip (d27 — most-shaped pit row to date)
  Environment: make_env_pit_d23 (pit_timing_reward=True) — same as d27
  Three-layer freeze: identical to d26/d27
    Layer 1: mlp_extractor.policy_net (requires_grad=False)
    Layer 2: action_net weight rows 0,1 + bias[0,1] (gradient hooks)
    Layer 3: log_std[0,1] (gradient hook)
  Only 130 policy params trainable: action_net.weight[2,:] + bias[2] + log_std[2]
  STAGES_PIT_V5: safety net 0.25→0.15→0.08
  LR: cosine 1e-4→1e-6 with reset_num_timesteps=True
  Total steps: 2M

WHY D27.ZIP INSTEAD OF D26.ZIP?
  D27's pit row has additional shaping from the pit_timing_reward gradient
  (even though small): weight 0.301→0.305, bias -0.817→-0.839.
  The pit row is already slightly adapted to the timing signal.
  Starting from d27 builds on this while adding proper LR.

EXPECTED OUTCOME:
  With LR=1e-4 (3x higher than d27's 3.5e-5):
  - The pit timing gradient (+100 for tyre_life < 0.30) should be strong enough
    to shift the pit row decision boundary from ≈0.35 toward ≈0.29-0.30.
  - Driving: IDENTICAL (three-layer freeze, drift = 0.00e+00)
  - Fixed-start reward: > 1883 (if timing shifts to earn the +100 bonus)
  - Random-start pits: ≥ 1.30 avg (d27's already high count maintained)

COMPARISON TO D27:
  d27: LR started at 3.5e-5 (wrong), pit row +1.1%, same fixed-start reward
  d28: LR starts at 1e-4 (correct), pit row should shift more, target +100 reward

STARTING POINT: ppo_pit_v4_d27.zip (d27, reward=1883, pits=1.30 avg random)
SAVES TO:       rl/ppo_pit_v4_d28.zip
LOGS TO:        runs/ppo_pit_v4_d28/
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

from rl.make_env import make_env_pit_d23      # pit_timing_reward=True
from rl.curriculum import CurriculumCallback, STAGES_PIT_V5
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d27.zip")

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4_d27.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4_d27.py (d27) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d27, reward=1883, pits=1.30 avg random)")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit_d23])

    # ── Load d27 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Pre-training diagnostics
    pit_w_before  = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_before  = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_before  = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_before = model.policy.action_net.bias[2].item()
    thr_std_before = model.policy.log_std[0].item()
    str_std_before = model.policy.log_std[1].item()
    pit_std_before = model.policy.log_std[2].item()
    print(f"\n[Diag] Pre-training weights (d27's pit row):")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_before:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_before:.6f}  (steer)")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_before:.6f}  (pit)")
    print(f"       action_net.bias[2]     = {pit_bias_before:.6f}  (pit bias)")
    print(f"       log_std[0] = {thr_std_before:.6f} (throttle std)")
    print(f"       log_std[1] = {str_std_before:.6f} (steer std)")
    print(f"       log_std[2] = {pit_std_before:.6f} (pit std)")

    # ── LAYER 1: Freeze mlp_extractor.policy_net ──────────────────────────────
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Freeze throttle/steer rows via gradient hooks ────────────────

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0, :] = 0.0
        g[1, :] = 0.0
        return g

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    print(f"[Freeze] Layer 2: action_net.weight rows [0,1] + bias[0,1] (gradient hooks)")

    # ── LAYER 3: Freeze log_std throttle/steer dims ──────────────────────────

    def _hook_log_std(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layer 3: log_std[0,1] (gradient hook)")

    # Print summary
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Freeze] Summary:")
    print(f"         Frozen (features):    {frozen_feat:,} params")
    print(f"         Frozen (hook):        action_net rows 0,1 + log_std[0,1]")
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 130 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── LR schedule ───────────────────────────────────────────────────────────
    # THE KEY FIX: cosine 1e-4→1e-6 starting from progress=1.0 (not 0.4).
    # reset_num_timesteps=True below ensures progress starts at 1.0.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure curriculum callback ─────────────────────────────────────────
    callback = CurriculumCallback(stages=STAGES_PIT_V5, verbose=1)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d28", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D28: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:       ppo_pit_v4_d27.zip (d27, reward=1883)")
    print(f"        Starting LR:         1e-4 (cosine decay to 1e-6 — FRESH START)")
    print(f"        Environment:         make_env_pit_d23 (pit_timing_reward=True)")
    print(f"        reset_num_timesteps: True (LR schedule starts at progress=1.0)")
    print(f"        Frozen (Layer 1):    mlp_extractor.policy_net ({frozen_feat:,} params)")
    print(f"        Frozen (Layer 2):    action_net rows 0,1 via hooks")
    print(f"        Frozen (Layer 3):    log_std[0,1] via hook")
    print(f"        Trainable policy:    130 params (pit row only)")
    print(f"        Curriculum:          STAGES_PIT_V5 (safety net: 0.25→0.15→0.08)")
    print(f"        Goal:                reward > 1883 AND pit at tyre_life < 0.30")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,    # THE KEY FIX: LR starts at 1e-4, not 3.5e-5
        callback=callback,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pit_w_after  = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_after  = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after  = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_after = model.policy.action_net.bias[2].item()
    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] = {thr_w_after:.6f}  (was {thr_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[1,:] = {str_w_after:.6f}  (was {str_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[2,:] = {pit_w_after:.6f}  (was {pit_w_before:.6f}) [PIT: can change]")
    print(f"       action_net.bias[2]     = {pit_bias_after:.6f}  (was {pit_bias_before:.6f}) [PIT: can change]")

    # Freeze verification vs d27
    state_d27 = PPO.load(checkpoint_path, device=device)

    feature_drift = max(
        (p_new - p_old).abs().max().item()
        for p_new, p_old in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            state_d27.policy.mlp_extractor.policy_net.parameters()
        )
    )
    thr_drift = (
        model.policy.action_net.weight[0, :] - state_d27.policy.action_net.weight[0, :]
    ).abs().max().item()
    str_drift = (
        model.policy.action_net.weight[1, :] - state_d27.policy.action_net.weight[1, :]
    ).abs().max().item()

    print(f"\n[Diag] Freeze verification vs d27 (all should be ≈ 0.0):")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features]")
    print(f"       action_net.weight[0,:] drift   = {thr_drift:.2e}  [throttle row]")
    print(f"       action_net.weight[1,:] drift   = {str_drift:.2e}  [steer row]")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d28.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d28 model to {save_path}")
    print(f"        Run evaluate.py to compare against d27 (1883, 1.30 avg pits).")
    print(f"        If reward > 1883: timing shifted to earn +100 bonus.")
    print(f"        If reward ≈ 1883 but pit_bias shifted toward 0: partial progress.")


if __name__ == "__main__":
    train()
