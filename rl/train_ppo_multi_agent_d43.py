"""
PPO Multi-Agent Racing D43 — Intermediate Opponent at 25 m/s (Week 6).

WHY D43? FINDING THE SWEET SPOT
==================================
D39 (opp=22 m/s): col[11/12] weights = 0. Raw speed advantage (26.9 > 22) →
  no positional awareness needed. Result: 6025 reward, 17 laps, 100% ✓
  BUT: no strategic behavior.

D41 (opp=27 m/s): col[11] weight = 0.016 (non-zero ✓). Shortcut closed →
  agent began using track_gap. HOWEVER: eval = 69.5 reward, 0 laps, 0% ✗
  Equal-speed competition too hard — policy relied on stochastic exploration.

D43 hypothesis (opp=25 m/s):
  - 2 m/s speed advantage (~8%) closes the "ignore opponent" shortcut
  - Small but real edge allows reliable lap completion
  - Agent inherits D41's non-zero track_gap weights → faster convergence
  - Expected: col[11] weight non-zero AND ≥100% completion in eval

SUCCESS CRITERIA:
  - col[11] (track_gap) weight > 0.005 (maintained from D41)
  - Fixed-start reward ≥ 3000 (lap completion restored)
  - Fixed-start laps ≥ 10
  - Fixed-start completion = 100%

TRAINING SETUP:
  Start:  ppo_multi_agent_d41.zip (13D, 2D action — no obs extension needed)
  Steps:  3M
  LR:     cosine 1e-4 → 1e-6
  Env:    F1MultiAgentEnv(opp_max_speed=25.0)
  ent_coef: 0.01 (prevent log_std collapse — d38 lesson)

SAVES TO: rl/ppo_multi_agent_d43.zip
LOGS TO:  runs/ppo_multi_agent_d43/
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

from rl.make_env import make_env_multi_agent_d43
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_multi_agent_d41.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_agent_d41.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_agent_d41.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D41: 3170 ep_rew (training), col[11]=0.0158, col[12]=0.0048 — opp=27 m/s")

    # ── Load D41 (13D) ────────────────────────────────────────────────────────
    print(f"\n[Train] Loading ppo_multi_agent_d41...")
    model = PPO.load(checkpoint_path, device=device)

    # Capture pre-training diagnostics
    pnet0_before   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_before   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_before = model.policy.log_std.tolist()
    tg_col_before  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_before = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] D41 loaded weights:")
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
    # D41 was trained for 3M steps against opp=27 m/s — its Adam momentum
    # encodes the gradient history from that harder task. Resetting gives a
    # clean start relative to the easier opp=25 m/s environment.
    model.policy.optimizer = optim.Adam(
        model.policy.parameters(), lr=1e-4, eps=1e-5
    )
    print(f"[Optimizer] Adam optimizer reset (fresh — avoids stale D41 momentum)")

    # ── Set environment (opponent at 25 m/s) ──────────────────────────────────
    env = DummyVecEnv([make_env_multi_agent_d43] * N_ENVS)
    model.set_env(env)
    print(f"[Env] F1MultiAgentEnv(opp_max_speed=25.0) — 2 m/s ego advantage")

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

    model.set_logger(configure("runs/ppo_multi_agent_d43", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent D43 — Ego vs ExpertDriver (25 m/s): {TOTAL_STEPS:,} steps")
    print(f"        Starting: D41 (3170 ep_rew training, col[11]=0.016)")
    print(f"        Goal: col[11] > 0 AND 100% lap completion AND reward ≥ 3000")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()
    tg_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs D41):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [{(pnet0_after-pnet0_before)/pnet0_before*100:+.1f}%]")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [{(pnet2_after-pnet2_before)/pnet2_before*100:+.1f}%]")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_after:.6f}  [D41={tg_col_before:.6f}]")
    print(f"       col[12] (opp_speed)    abs_mean = {osp_col_after:.6f}  [D41={osp_col_before:.6f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    positional = "YES ✓" if tg_col_after > 0.005 else "NO ✗"
    print(f"\n[Result] Does agent use track_gap? {positional}")
    print(f"         col[11] D41={tg_col_before:.6f} → D43={tg_col_after:.6f}")

    save_path = str(project_root / "rl" / "ppo_multi_agent_d43.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d43 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")


if __name__ == "__main__":
    train()
