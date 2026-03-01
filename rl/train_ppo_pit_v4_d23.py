"""
PPO Pit Strategy v4 D23 — Pit Reward Shaping + Frozen Pit Row (Week 5).

WHY D23?
=========
d22 showed a critical problem: 2M fine-tuning steps from d21 caused the agent
to FORGET pitting entirely (pit_signal → -1.0, 0 pits) by finding a better
local optimum — drive perfectly without pitting (reward=2283, 0.182 m lateral).

ROOT CAUSE (d22 post-mortem):
  The -200 PIT_PENALTY was avoidable by driving better. The agent discovered:
    "drive well, skip pit" → +406 more reward than d21's "pit once, drive OK"
  Without an explicit signal for correct pit timing, the agent rationally
  chose to never pit. PPO's on-policy updates overwrote the pit knowledge
  completely over 2M steps. pit_signal collapsed back to -1.0000 (same as
  the pre-discovery d18/d19 baseline).

TWO-PART FIX:
  Part A — Pit Reward Shaping (env/f1_env.py + make_env.py):
    pit_timing_reward=True in F1Env enables explicit timing incentives:
      tyre_life < 0.3 (worn, correct timing):  net pit cost = -100 (was -200)
      tyre_life > 0.5 (fresh, wrong timing):   net pit cost = -300 (was -200)
    This makes correct pit timing explicitly rewarded, not a lucky side effect.
    The agent now has a POSITIVE reason to pit on worn tyres (not just "avoid -200").

  Part B — Freeze Pit Row During Fine-Tuning:
    Register PyTorch backward hooks that zero out gradients for:
      action_net.weight[2, :]  — pit signal mean output weights  (latent_dim values)
      action_net.bias[2]       — pit signal mean output bias     (1 value)
      log_std[2]               — pit signal exploration scale    (1 value)
    These hooks fire during every backward pass. The optimizer sees zero gradient
    for row 2, so Adam does NOT update those parameters. The pit policy from
    ppo_pit_v4 (d21) is LOCKED — it cannot be overwritten by fine-tuning.

    Why hooks instead of requires_grad=False?
      PyTorch doesn't support requires_grad=False on a slice of a parameter.
      (Setting it on the whole parameter would also freeze throttle/steer rows.)
      Backward hooks let us selectively zero the gradient for specific rows
      while leaving rows 0 (throttle) and 1 (steer) free to update normally.

    Why not freeze the entire pit MLP path?
      The hidden layers (mlp_extractor) feed into all three output rows.
      If we froze the shared hidden layers, driving quality couldn't improve.
      Freezing only the pit OUTPUT row lets driving improve while preserving
      the pit DECISION. The shared representation can still reorganize for
      better driving; the pit head simply won't be overwritten.

STARTING POINT: ppo_pit_v4.zip (d21) — NOT ppo_pit_v4_cont.zip (d22).
  d21: pit_signal fires once at right time (pit_count=1, pit signal frac > 0 = 0.001)
  d22: pit_signal LOST (pit_signal mean=-1.0000, pit_count=0, behavior forgotten)
  We MUST restart from d21 to preserve the pit behavior that d22 destroyed.

EXPECTED OUTCOME:
  - Pit behavior preserved: frozen pit row prevents the forgetting seen in d22
  - Correct pit timing rewarded: +100 bonus for tyre_life < 0.3 pit signals
  - Driving improves: rows 0 (throttle) and 1 (steer) continue to learn
  - Combined result: reward > 2283 (d22) with pit_count > 0

SAVES TO: rl/ppo_pit_v4_d23.zip
LOGS TO:  runs/ppo_pit_v4_d23/
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

from rl.make_env import make_env_pit_d23
from rl.schedules import cosine_schedule


# ── Gradient hooks: freeze the pit output row ────────────────────────────────
# We register hooks on each relevant parameter tensor. Each hook receives the
# full gradient tensor and returns a modified version with row 2 zeroed out.
# The optimizer then sees zero gradient for those entries → no weight update.

def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
    """Zero gradient for action_net.weight row 2 (pit mean weights)."""
    g = grad.clone()
    g[2, :] = 0.0   # pit signal row: latent_dim weights → all zeroed
    return g


def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
    """Zero gradient for action_net.bias[2] AND log_std[2] (both 1D tensors)."""
    g = grad.clone()
    g[2] = 0.0      # pit signal entry: 1 scalar value → zeroed
    return g


def freeze_pit_row(model: PPO) -> None:
    """
    Register backward hooks to freeze the pit output row (index 2).

    After this call, every backward pass zeros out the gradient contribution
    to the pit dimension of the actor network. The optimizer (Adam) then has
    nothing to update for those entries.

    WHAT IS FROZEN:
      model.policy.action_net.weight[2, :]  — pit mean output weights
      model.policy.action_net.bias[2]       — pit mean output bias
      model.policy.log_std[2]               — pit exploration scale

    WHAT REMAINS TRAINABLE:
      model.policy.action_net.weight[0, :]  — throttle mean output
      model.policy.action_net.weight[1, :]  — steer mean output
      model.policy.action_net.bias[0]       — throttle mean bias
      model.policy.action_net.bias[1]       — steer mean bias
      model.policy.log_std[0]               — throttle exploration scale
      model.policy.log_std[1]               — steer exploration scale
      model.policy.mlp_extractor.*          — all shared hidden layers (free to update)
      model.policy.value_net.*              — value head (free to update)

    Args:
        model: Loaded SB3 PPO model with 3D action space (throttle, steer, pit).
    """
    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    model.policy.log_std.register_hook(_hook_bias)   # same shape (3,): reuse scalar hook

    print("[Freeze] Pit row (index 2) of action_net + log_std frozen via gradient hooks.")
    print(f"         action_net: shape {tuple(model.policy.action_net.weight.shape)}")
    print(f"         log_std:    shape {tuple(model.policy.log_std.shape)}")
    print("         Rows [0]=throttle, [1]=steer remain fully trainable.")


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    ADDITIONAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")   # d21 — NOT d22!

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21 weights, pit behavior intact)")

    # ── Build environment with pit timing reward shaping ─────────────────────
    # make_env_pit_d23() uses F1Env(pit_timing_reward=True):
    #   Correct pit timing (tyre_life < 0.3) → net cost -100 (was -200)
    #   Wrong pit timing  (tyre_life > 0.5)  → net cost -300 (was -200)
    env = DummyVecEnv([make_env_pit_d23])

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded. Pit behavior from d21 is intact.")

    # Diagnostic: verify the pit row hasn't already collapsed
    pit_weights = model.policy.action_net.weight[2, :].abs().mean().item()
    pit_bias    = model.policy.action_net.bias[2].item()
    pit_log_std = model.policy.log_std[2].item()
    print(f"\n[Freeze] Pre-hook pit row diagnostics (should be non-trivial for d21):")
    print(f"         action_net.weight[2,:] abs_mean = {pit_weights:.6f}")
    print(f"         action_net.bias[2]              = {pit_bias:.6f}")
    print(f"         log_std[2]                      = {pit_log_std:.6f}")

    # ── Register gradient hooks to freeze pit row ─────────────────────────────
    # MUST be done BEFORE model.learn() so hooks are active from step 1.
    freeze_pit_row(model)

    # ── Lower LR for fine-tuning ───────────────────────────────────────────────
    # Same LR schedule as d22 (fine-tuning mode, not discovery).
    # Only throttle/steer rows will be updated — smaller LR reduces risk of
    # destabilizing the driving policy that's already good from d21.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d23", ["stdout", "tensorboard"]))

    # ── Run continued training ─────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D23: {ADDITIONAL_STEPS:,} steps")
    print(f"        Starting from:    ppo_pit_v4.zip (d21, pit_count=1, reward=1877)")
    print(f"        Starting LR:      1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        Environment:      make_env_pit_d23 (pit_timing_reward=True)")
    print(f"        Frozen:           action_net pit row (index 2) + log_std[2]")
    print(f"        Trainable:        throttle/steer rows (0,1) + hidden layers + value head")
    print(f"        Pit reward:       tyre_life<0.3 → net -100, tyre_life>0.5 → net -300")
    print(f"        Goal:             reward > 2283 (d22) AND pit_count > 0")
    print(f"        TensorBoard:      tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False,   # step counter continues from d21's ~1M
    )

    # ── Diagnostic: verify pit row is unchanged ────────────────────────────────
    pit_weights_after = model.policy.action_net.weight[2, :].abs().mean().item()
    pit_bias_after    = model.policy.action_net.bias[2].item()
    pit_log_std_after = model.policy.log_std[2].item()
    print(f"\n[Freeze] Post-training pit row diagnostics (should match pre-hook values):")
    print(f"         action_net.weight[2,:] abs_mean = {pit_weights_after:.6f}  (was {pit_weights:.6f})")
    print(f"         action_net.bias[2]              = {pit_bias_after:.6f}  (was {pit_bias:.6f})")
    print(f"         log_std[2]                      = {pit_log_std_after:.6f}  (was {pit_log_std:.6f})")

    weight_drift = abs(pit_weights_after - pit_weights)
    if weight_drift > 1e-6:
        print(f"[WARN] Pit weight drift = {weight_drift:.2e} — hooks may not have fully prevented updates.")
    else:
        print(f"[OK] Pit row unchanged. Gradient hooks worked correctly.")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d23.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d23 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (reward=1877) and d22 (reward=2283).")


if __name__ == "__main__":
    train()
