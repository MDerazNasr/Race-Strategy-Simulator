"""
PPO Multi-Agent Racing D44 — Stronger Position Bonus (Week 6).

WHY D44? FIXING THE FOLLOW EQUILIBRIUM
=========================================
D43 (opp=25 m/s, pos_bonus=0.5) revealed a "follow closely" local optimum:
  - Training ep_rew = 3390, ep_len = 1170 steps (moderate, some crashes)
  - Deterministic eval speed = 24.36 m/s < opponent 25 m/s
  - col[11] (track_gap) weight = 0.022 (positional awareness exists)
  - BUT eval: 0 laps, 275 reward, 0% completion

Root cause: with position_bonus=0.5/step, the 2000-step value of being ahead
is +1000. The agent found a "follow at ~opponent speed" equilibrium that
yields ~0 position bonus but also avoids collision risk. In deterministic eval,
this equilibrium is stable (no noise to perturb it), so the ego never overtakes.

Fix: raise position_bonus to 2.0/step.
  Value of being ahead: 0.5 * 2000 = +1000  →  2.0 * 2000 = +4000
  The policy must learn aggressive overtaking to capture this 4x larger reward.
  D43's non-zero track_gap weight means the agent already knows WHAT to look at —
  it just needs a stronger incentive to ACT on it.

SUCCESS CRITERIA:
  - col[11] (track_gap) weight > 0.015 (maintained or improved from D43)
  - Fixed-start reward ≥ 3000
  - Fixed-start laps ≥ 10
  - Fixed-start completion = 100%

TRAINING SETUP:
  Start:  ppo_multi_agent_d43.zip (13D, 2D action — no obs extension needed)
  Steps:  3M
  LR:     cosine 1e-4 → 1e-6
  Env:    F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)
  ent_coef: 0.01

SAVES TO: rl/ppo_multi_agent_d44.zip
LOGS TO:  runs/ppo_multi_agent_d44/
"""

import sys
import math
from pathlib import Path

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

from rl.make_env import make_env_multi_agent_d44
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_multi_agent_d43.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_agent_d43.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_agent_d43.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D43: 3390 ep_rew (training), col[11]=0.0216 — opp=25 m/s, pos_bonus=0.5")

    # ── Load D43 (13D) ────────────────────────────────────────────────────────
    print(f"\n[Train] Loading ppo_multi_agent_d43...")
    model = PPO.load(checkpoint_path, device=device)

    # Pre-training diagnostics
    pnet0_before   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_before   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_before = model.policy.log_std.tolist()
    tg_col_before  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_before = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] D43 loaded weights:")
    print(f"       policy_net[0] abs_mean = {pnet0_before:.6f}")
    print(f"       policy_net[2] abs_mean = {pnet2_before:.6f}")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_before:.6f}")
    print(f"       col[12] (opp_speed)    abs_mean = {osp_col_before:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_before)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_before)}]")

    # ── Recreate rollout buffer ────────────────────────────────────────────────
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
    # Reward scale changed (position_bonus 0.5 → 2.0): the value function
    # must re-calibrate. Fresh optimizer prevents stale gradients from the
    # old reward scale from interfering with early adaptation.
    model.policy.optimizer = optim.Adam(
        model.policy.parameters(), lr=1e-4, eps=1e-5
    )
    print(f"[Optimizer] Adam optimizer reset (reward scale changed → fresh momentum)")

    # ── Set environment ────────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_multi_agent_d44] * N_ENVS)
    model.set_env(env)
    print(f"[Env] F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)")
    print(f"      Value of being ahead (full episode): 0.5*2000=1000 → 2.0*2000=4000")

    # ── LR schedule ───────────────────────────────────────────────────────────
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    model.ent_coef = ENT_COEF
    print(f"[Entropy] ent_coef = {ENT_COEF}")

    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")

    model.set_logger(configure("runs/ppo_multi_agent_d44", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent D44 — opp=25 m/s, pos_bonus=2.0: {TOTAL_STEPS:,} steps")
    print(f"        Starting: D43 (3390 ep_rew training, col[11]=0.022)")
    print(f"        Key change: position_bonus 0.5 → 2.0 (4x, breaks follow equilibrium)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()
    tg_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs D43):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [{(pnet0_after-pnet0_before)/pnet0_before*100:+.1f}%]")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [{(pnet2_after-pnet2_before)/pnet2_before*100:+.1f}%]")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_after:.6f}  [D43={tg_col_before:.6f}]")
    print(f"       col[12] (opp_speed)    abs_mean = {osp_col_after:.6f}  [D43={osp_col_before:.6f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    positional = "YES ✓" if tg_col_after > 0.015 else "NO ✗"
    print(f"\n[Result] Does agent use track_gap? {positional}")
    print(f"         col[11] D43={tg_col_before:.6f} → D44={tg_col_after:.6f}")

    save_path = str(project_root / "rl" / "ppo_multi_agent_d44.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d44 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")


if __name__ == "__main__":
    train()
