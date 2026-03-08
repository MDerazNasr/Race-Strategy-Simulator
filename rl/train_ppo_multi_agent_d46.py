"""
PPO Multi-Agent Racing D46 — Continue D45 (3M more steps).

WHY D46? LAP PROGRESSION
===========================
D44: 3 laps before crash (follow equilibrium broken, speed=26.91)
D45: 8 laps before crash (continued improvement, speed=27.18 — project fastest)
D46 hypothesis: 3M more steps → 13+ laps → reach 100% completion

The trajectory is clear: each 3M step run adds ~5 laps. The policy is
iteratively refining the critical moment when the ego (27+ m/s) approaches
the opponent (25 m/s) for a pass. More training = more stable passes.

TRAINING SETUP:
  Start:  ppo_multi_agent_d45.zip (13D, 2D action)
  Steps:  3M
  LR:     cosine 1e-5 → 1e-6  (even lower — very fine-tuning)
  Env:    F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)
  ent_coef: 0.005

SAVES TO: rl/ppo_multi_agent_d46.zip
LOGS TO:  runs/ppo_multi_agent_d46/
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

    checkpoint_path = str(project_root / "rl" / "ppo_multi_agent_d45.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_agent_d45.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_agent_d45.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D45: 3910 ep_rew, col[11]=0.067, speed=27.18, 8 laps before crash")

    model = PPO.load(checkpoint_path, device=device)

    tg_col_before = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    log_std_before = model.policy.log_std.tolist()
    print(f"\n[Diag] D45 loaded: col[11]={tg_col_before:.6f}, "
          f"std=[{', '.join(f'{math.exp(x):.4f}' for x in log_std_before)}]")

    model.rollout_buffer = RolloutBuffer(
        model.n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=N_ENVS,
    )

    # Very low LR — D45 already converged, minimal nudges only
    model.policy.optimizer = optim.Adam(model.policy.parameters(), lr=1e-5, eps=1e-5)

    env = DummyVecEnv([make_env_multi_agent_d44] * N_ENVS)
    model.set_env(env)
    print(f"[Env] F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0) — unchanged")

    new_lr = cosine_schedule(initial_lr=1e-5, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    model.ent_coef = ENT_COEF

    for param in model.policy.parameters():
        param.requires_grad = True

    model.set_logger(configure("runs/ppo_multi_agent_d46", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent D46 — continue D45: {TOTAL_STEPS:,} steps")
    print(f"        LR: 1e-5 → 1e-6 (very fine-tuning)")
    print(f"        Goal: 13+ laps → 100% lap completion\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    tg_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()
    log_std_after = model.policy.log_std.tolist()

    print(f"\n[Diag] Post-training:")
    print(f"       col[11] (track_gap) = {tg_col_after:.6f}  [D45={tg_col_before:.6f}]")
    print(f"       col[12] (opp_speed) = {osp_col_after:.6f}")
    print(f"       std = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    save_path = str(project_root / "rl" / "ppo_multi_agent_d46.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d46 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")


if __name__ == "__main__":
    train()
