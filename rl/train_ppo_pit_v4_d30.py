"""
PPO Pit Strategy v4 D30 — Voluntary Pit Reward Shaping (Week 5).

WHY D30?
=========
D26/D27/D28/D29 all failed to break the 1883 ceiling because the forced-pit
curriculum trains pit_signal in the WRONG direction:

  Forced pit fires regardless of agent's pit_signal.
  Agent outputs pit_signal < 0 (negative bias).
  Forced pit fires → agent survives longer → POSITIVE advantage.
  PPO credits that advantage to the CHOSEN action: pit_signal < 0.
  Gradient reinforces: "low pit_signal at worn tyre → positive outcome"
  Bias: d21=+0.006 → d26=-0.817 → d28=-0.929 → d29=-1.217 (consistently wrong)

THE D30 FIX — VOLUNTARY PIT REWARD:
  Remove forced pits entirely. Instead, give a +300 bonus when the agent
  CHOOSES to pit (pit_signal > 0) at tyre_life < 0.60 (voluntary_pit_reward).

  Net voluntary pit cost: -200 (penalty) + 300 (bonus) = +100.
  Plus: fresh tyres → survive to step 2000 → ~+800 more reward.
  TOTAL advantage of voluntary pit: ~+900 vs crashing at step 1354.

  The gradient now sees: (worn_tyre_state, pit_signal > 0) → high reward.
  This directly reinforces pit_signal > 0 at worn-tyre states.

EXPLORATION MATH:
  D21 pit_std = exp(log_std[2]=0.76) ≈ 2.13.
  D21 pit_bias ≈ +0.006 (near-zero).
  P(pit_signal > 0) ≈ 50% per step with Gaussian N(0, 2.13).
  tyre_life < 0.60 at step ~571 (fixed-start trajectory).
  → Agent will discover the voluntary pit bonus within first few episodes.

LR BUG FIX (from d29 post-mortem):
  d29 set model.learning_rate but NOT model.lr_schedule.
  SB3 uses model.lr_schedule for optimizer updates.
  Fix: set BOTH model.learning_rate AND model.lr_schedule.

STARTING POINT: ppo_pit_v4.zip (d21, reward=1877, 7 laps, 1 pit, bias=+0.006)
SAVES TO:       rl/ppo_pit_v4_d30.zip
LOGS TO:        runs/ppo_pit_v4_d30/
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

from rl.make_env import make_env_pit_d30       # voluntary_pit_reward=True, no forced pit
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21, reward=1877, bias=+0.006)")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit_d30])    # voluntary_pit_reward=True, no forced pit

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Pre-training diagnostics
    pit_w_before    = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_before    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_before    = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_before = model.policy.action_net.bias[2].item()
    thr_std_before  = model.policy.log_std[0].item()
    str_std_before  = model.policy.log_std[1].item()
    pit_std_before  = model.policy.log_std[2].item()
    print(f"\n[Diag] Pre-training weights (d21 pit row):")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_before:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_before:.6f}  (steer)")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_before:.6f}  (pit)")
    print(f"       action_net.bias[2]     = {pit_bias_before:.6f}  (pit bias — d21 near-zero)")
    print(f"       log_std[0] = {thr_std_before:.6f} (throttle std)")
    print(f"       log_std[1] = {str_std_before:.6f} (steer std)")
    print(f"       log_std[2] = {pit_std_before:.6f} (pit std, exp={torch.exp(torch.tensor(pit_std_before)).item():.3f})")
    print(f"\n[Diag] P(pit_signal > 0) ≈ 50% with bias={pit_bias_before:+.4f} and std={torch.exp(torch.tensor(pit_std_before)).item():.2f}")

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

    # ── LR schedule — FIX D29 BUG: set BOTH learning_rate AND lr_schedule ─────
    # D29 bug: model.learning_rate = new_schedule only changes the attribute.
    # SB3 uses model.lr_schedule for actual optimizer updates.
    # After PPO.load(), lr_schedule points to d21's old closure (initial=3e-4).
    # Fix: update BOTH so they're in sync.
    new_schedule = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_schedule
    model.lr_schedule   = new_schedule    # ← THE FIX (was missing in d29)
    print(f"\n[LR] Set model.learning_rate AND model.lr_schedule = cosine(1e-4 → 1e-6)")
    print(f"     D29 bug fixed: SB3 now uses 1e-4 start (not d21's 3e-4)")

    # Verify: LR at progress=1.0 should be 1e-4
    lr_at_start = new_schedule(1.0)
    print(f"     Confirmed: lr_schedule(1.0) = {lr_at_start:.2e}  (should be 1e-4)")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d30", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D30: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:       ppo_pit_v4.zip (d21, reward=1877, bias=+0.006)")
    print(f"        Starting LR:         1e-4 (cosine decay to 1e-6 — FIXED from d29)")
    print(f"        Environment:         make_env_pit_d30 (voluntary_pit_reward=True)")
    print(f"        voluntary_pit_bonus: +300 when pit_signal>0 AND tyre_life<0.60")
    print(f"        Net voluntary pit:   -200+300 = +100  (PROFITABLE to pit voluntarily!)")
    print(f"        No forced pit:       agent must discover pitting via exploration")
    print(f"        reset_num_timesteps: True")
    print(f"        Frozen (Layer 1):    mlp_extractor.policy_net ({frozen_feat:,} params)")
    print(f"        Frozen (Layer 2):    action_net rows 0,1 via hooks")
    print(f"        Frozen (Layer 3):    log_std[0,1] via hook")
    print(f"        Trainable policy:    130 params (pit row only)")
    print(f"        No curriculum:       voluntary reward replaces forced pit")
    print(f"        Goal: pit bias → positive; voluntary pit at tyre_life<0.60")
    print(f"              reward > 2000 on fixed-start (pit fires before step 1354)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,    # LR starts at 1e-4 (fresh schedule)
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pit_w_after    = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_after    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after    = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_after = model.policy.action_net.bias[2].item()
    pit_std_after  = model.policy.log_std[2].item()
    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] = {thr_w_after:.6f}  (was {thr_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[1,:] = {str_w_after:.6f}  (was {str_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[2,:] = {pit_w_after:.6f}  (was {pit_w_before:.6f}) [PIT: can change]")
    print(f"       action_net.bias[2]     = {pit_bias_after:.6f}  (was {pit_bias_before:.6f}) [PIT: can change]")
    print(f"       log_std[2]             = {pit_std_after:.6f}  (was {pit_std_before:.6f}) [PIT: can change]")

    # Freeze verification vs d21
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

    print(f"\n[Diag] Freeze verification vs d21 (all should be ≈ 0.0):")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features]")
    print(f"       action_net.weight[0,:] drift   = {thr_drift:.2e}  [throttle row]")
    print(f"       action_net.weight[1,:] drift   = {str_drift:.2e}  [steer row]")

    # Key diagnosis: did pit bias go POSITIVE?
    print(f"\n[Diag] Pit bias trajectory (d21→d30):")
    print(f"       d21 (start): {pit_bias_before:+.6f}")
    print(f"       d30 (end):   {pit_bias_after:+.6f}")
    if pit_bias_after > 0:
        print(f"       → POSITIVE BIAS (+{pit_bias_after:.4f}) — agent WANTS to pit!")
        print(f"       → P(pit_signal > 0) at worn tyre ≈ high (voluntary pitting learned!)")
    elif pit_bias_after > pit_bias_before:
        print(f"       → POSITIVE SHIFT (+{pit_bias_after - pit_bias_before:.4f}) — moving right direction!")
    else:
        print(f"       → NEGATIVE SHIFT ({pit_bias_after - pit_bias_before:.4f}) — investigate (voluntary reward not working?)")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d30.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d30 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (1877) and d29 (1885).")
    print(f"        If pit_bias > 0: voluntary pit discovered. Check fixed-start reward.")
    print(f"        Target: reward > 2000 (pit before step 1354, survive to step 2000).")


if __name__ == "__main__":
    train()
