"""
PPO Multi-Agent Racing D41 — Competitive Opponent at 27 m/s (Week 6).

WHY D41? CLOSING THE SPEED SHORTCUT
======================================
D39 (ego vs ExpertDriver at 22 m/s) revealed a critical finding:
  - col[11] (track_gap) weight     = 0
  - col[12] (opp_speed_norm) weight = 0
  - Agent simply drove faster (26.9 > 22 m/s) and lapped the opponent
  - No strategic positioning, blocking, or conditional behavior emerged

Pattern: PPO finds the SIMPLEST strategy that works.
With a 4.9 m/s speed advantage, that strategy is "go fast, ignore opponent."

D41 closes this shortcut by raising the opponent speed to 27.0 m/s —
matching (and slightly exceeding) the ego's cruising speed from D39.
The ego can no longer rely on raw speed advantage; it must use
track_gap to know its position and adjust behavior accordingly.

EXPECTED OUTCOMES:
  - col[11] (track_gap) weight > 0 — agent uses positional awareness
  - Agent develops defensive racing (staying ahead when challenged)
  - Possibly: blocking behavior when opponent approaches
  - Possibly: more aggressive throttle on straights to create gaps

SUCCESS CRITERIA:
  - col[11] (track_gap) weight != 0 in policy_net[0]
  - Ego spends >50% of time ahead on fixed start
  - Fixed-start reward ≥ 2500 (driving quality maintained)
  - At least 1 successful overtake per fixed-start episode on average

TRAINING SETUP:
  Start:  ppo_multi_agent_d39.zip (13D, 2D action — no obs extension needed)
  Steps:  3M
  LR:     cosine 1e-4 → 1e-6
  Env:    F1MultiAgentEnv(opp_max_speed=27.0)
  Envs:   1 parallel
  Freeze: NONE — full unfreeze

SAVES TO: rl/ppo_multi_agent_d41.zip
LOGS TO:  runs/ppo_multi_agent_d41/
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_multi_agent_d41
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_multi_agent_d39.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_agent_d39.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_agent_d39.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D39: 6025 reward, 17 laps, 26.91 m/s — multi-agent baseline (opp=22 m/s)")

    # ── Load D39 (13D) ────────────────────────────────────────────────────────
    print(f"\n[Train] Loading ppo_multi_agent_d39...")
    model = PPO.load(checkpoint_path, device=device)

    # Capture pre-training diagnostics
    pnet0_before = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_before = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_before = model.policy.log_std.tolist()
    tg_col_before  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_before = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] D39 loaded weights:")
    print(f"       policy_net[0] abs_mean = {pnet0_before:.6f}")
    print(f"       policy_net[2] abs_mean = {pnet2_before:.6f}")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_before:.6f}")
    print(f"       col[12] (opp_speed)    abs_mean = {osp_col_before:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_before)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_before)}]")

    # ── Recreate rollout buffer ────────────────────────────────────────────────
    # PPO.load pre-allocates the rollout buffer with the saved state.
    # Recreate to ensure clean slate with correct obs shape.
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=N_ENVS,
    )
    print(f"[Buffer] Rollout buffer recreated: obs_shape={model.observation_space.shape}")

    # ── Reset optimizer ────────────────────────────────────────────────────────
    # D39 was warm-started from cv2 (11D → extended to 13D). The Adam optimizer
    # exp_avg tensors were saved with shape (128, 11) from the original cv2 load,
    # not (128, 13) as the current weights require. This causes a shape mismatch
    # on the first Adam step. Fix: recreate the optimizer fresh.
    model.policy.optimizer = optim.Adam(
        model.policy.parameters(), lr=1e-4, eps=1e-5
    )
    print(f"[Optimizer] Adam optimizer reset (fresh — avoids stale 11D exp_avg from D39 lineage)")

    # ── Set environment (opponent at 27 m/s) ──────────────────────────────────
    env = DummyVecEnv([make_env_multi_agent_d41] * N_ENVS)
    model.set_env(env)
    print(f"[Env] F1MultiAgentEnv(opp_max_speed=27.0) — opponent matches ego top speed")

    # ── Learning rate schedule ─────────────────────────────────────────────────
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # ── Entropy coefficient ────────────────────────────────────────────────────
    model.ent_coef = ENT_COEF
    print(f"[Entropy] ent_coef = {ENT_COEF}")

    # ── Full unfreeze ──────────────────────────────────────────────────────────
    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")

    model.set_logger(configure("runs/ppo_multi_agent_d41", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent D41 — Ego vs ExpertDriver (27 m/s): {TOTAL_STEPS:,} steps")
    print(f"        Starting: D39 (6025 reward, 17 laps, 26.91 m/s)")
    print(f"        Challenge: opponent at 27 m/s — raw speed no longer sufficient")
    print(f"        Watch col[11] (track_gap) weight — should become non-zero")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()
    tg_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs D39):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [{(pnet0_after-pnet0_before)/pnet0_before*100:+.1f}%]")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [{(pnet2_after-pnet2_before)/pnet2_before*100:+.1f}%]")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_after:.6f}  [D39={tg_col_before:.6f}]")
    print(f"       col[12] (opp_speed)    abs_mean = {osp_col_after:.6f}  [D39={osp_col_before:.6f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    strategic_signal = "YES ✓" if tg_col_after > 2 * tg_col_before else "NO ✗"
    print(f"\n[Result] Did agent learn to use track_gap? {strategic_signal}")
    print(f"         col[11] D39={tg_col_before:.6f} → D41={tg_col_after:.6f}")

    save_path = str(project_root / "rl" / "ppo_multi_agent_d41.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d41 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")


if __name__ == "__main__":
    train()
