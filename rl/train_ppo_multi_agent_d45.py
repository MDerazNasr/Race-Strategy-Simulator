"""
PPO Multi-Agent Racing D45 — Continue D44, Same Reward (Week 6).

WHY D45? STABILITY FROM MORE TRAINING
========================================
D44 (opp=25 m/s, pos_bonus=2.0) broke the follow equilibrium:
  - Training: ep_rew=3820, ep_len=1040 (converged at 3M steps)
  - Eval: speed=26.91 m/s (FULL SPEED restored), 3 laps, 1834 reward
  - col[11] weight surged +155% (0.022 → 0.055): strongest positional signal

Remaining problem: crashes after ~3 laps (0% completion).
The reward signal is correct — the agent just needs more training to:
  1. Stabilize the overtaking maneuver (don't go off-track passing the opponent)
  2. Learn to maintain track discipline at 26.91 m/s with an active opponent

TRAINING SETUP:
  Start:  ppo_multi_agent_d44.zip (13D, 2D action)
  Steps:  3M
  LR:     cosine 3e-5 → 1e-6  (lower start — D44 already converged, fine-tune)
  Env:    F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)  (unchanged)
  ent_coef: 0.005  (slightly lower — D44 std=0.79 is healthy, preserve it)

SAVES TO: rl/ppo_multi_agent_d45.zip
LOGS TO:  runs/ppo_multi_agent_d45/
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
    ENT_COEF    = 0.005
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_multi_agent_d44.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_agent_d44.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_agent_d44.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D44: 3820 ep_rew, col[11]=0.055, speed=26.91, 3 laps before crash")

    model = PPO.load(checkpoint_path, device=device)

    pnet0_before   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_before   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_before = model.policy.log_std.tolist()
    tg_col_before  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_before = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] D44 loaded weights:")
    print(f"       policy_net[0] abs_mean = {pnet0_before:.6f}")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_before:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_before)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_before)}]")

    model.rollout_buffer = RolloutBuffer(
        model.n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=N_ENVS,
    )

    # Lower LR for fine-tuning — D44 already converged, avoid disrupting good weights
    model.policy.optimizer = optim.Adam(model.policy.parameters(), lr=3e-5, eps=1e-5)

    env = DummyVecEnv([make_env_multi_agent_d44] * N_ENVS)
    model.set_env(env)
    print(f"[Env] F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0) — unchanged")

    new_lr = cosine_schedule(initial_lr=3e-5, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    model.ent_coef = ENT_COEF
    print(f"[LR] cosine(3e-5 → 1e-6)  [lower — fine-tuning a converged policy]")
    print(f"[Entropy] ent_coef={ENT_COEF}  [lower — preserve healthy std=0.79]")

    for param in model.policy.parameters():
        param.requires_grad = True

    model.set_logger(configure("runs/ppo_multi_agent_d45", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent D45 — fine-tune D44: {TOTAL_STEPS:,} steps")
    print(f"        Goal: stabilize overtaking, reach 100% lap completion")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    pnet0_after   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()
    tg_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs D44):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [{(pnet0_after-pnet0_before)/pnet0_before*100:+.1f}%]")
    print(f"       col[11] (track_gap)    abs_mean = {tg_col_after:.6f}  [D44={tg_col_before:.6f}]")
    print(f"       col[12] (opp_speed)    abs_mean = {osp_col_after:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    save_path = str(project_root / "rl" / "ppo_multi_agent_d45.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d45 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")


if __name__ == "__main__":
    train()
