"""
PPO Curriculum Training on Monaco Circuit — D42.

WHY D42? REAL TRACK GEOMETRY
===============================
All previous experiments (D1–D41) trained on a synthetic oval (radius=50 m,
constant curvature). The three curvature lookahead observations — obs[5] (near),
obs[6] (mid), obs[7] (far) — carry zero information on the oval because every
corner looks identical.

Monaco changes this completely:
  - 3248 m per lap (vs oval ~314 m)
  - Drastically varying curvature: long straight → tight hairpin → tunnel → chicane
  - The near/mid/far curvature signals now differ at every track section
  - The agent MUST use them to brake early for tight corners and floor it on straights

This is the first experiment where the obs design from Week 3 (multi-step curvature)
is truly tested.

TRAINING PIPELINE (same as d13 curriculum):
  1. Expert data collection on Monaco (generate_monaco_dataset)
  2. BC warm start: 11D obs → [128, 128] → 2D action (train_bc_monaco.py)
  3. 3-stage PPO curriculum:
     Stage 1 — Stability: max_accel=6.0, grad 50% lap completion
     Stage 2 — Speed:     max_accel=11.0, grad 30% lap completion
     Stage 3 — Full:      max_accel=15.0, run to budget

Key difference: Monaco max_steps=6000 (not 2000) to allow ~4 laps per episode.

SAVES TO: rl/ppo_monaco.zip
LOGS TO:  runs/ppo_monaco/
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_monaco
from rl.bc_init_policy import load_bc_weights_into_ppo
from rl.schedules import cosine_schedule
from rl.curriculum import CurriculumCallback, STAGES


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000

    # ── Check prerequisites ───────────────────────────────────────────────────
    bc_path   = str(project_root / "bc" / "bc_policy_monaco.pt")
    data_path = str(project_root / "bc" / "expert_data_monaco.npz")

    if not Path(bc_path).exists():
        print(f"[Setup] BC weights not found at {bc_path}. Collecting data + training BC...")

        if not Path(data_path).exists():
            print("[Setup] Collecting Monaco expert data (50 episodes)...")
            from expert.collect_data import generate_monaco_dataset
            generate_monaco_dataset(num_episodes=50, max_steps=6000)

        print("[Setup] Training BC on Monaco data...")
        from bc.train_bc_monaco import train_bc_monaco
        train_bc_monaco()

    print(f"[Train] Starting D42 — Monaco PPO Curriculum")
    print(f"        BC weights: {bc_path}")

    # ── Create PPO model from scratch ─────────────────────────────────────────
    # Monaco env: 11D obs, 2D action — same as standard env.
    # Make a fresh PPO then load BC weights into the actor.
    env_fn   = make_env_monaco
    vec_env  = DummyVecEnv([env_fn])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),
        ent_coef=0.005,
        clip_range=0.1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device=device,
    )

    # Load BC weights into the actor (same utility as all previous experiments)
    load_bc_weights_into_ppo(model, bc_path, device=device)
    print(f"[BC Init] Loaded BC weights from {bc_path}")

    # ── Curriculum callback ───────────────────────────────────────────────────
    curriculum_cb = CurriculumCallback(stages=STAGES, verbose=True)

    model.set_logger(configure("runs/ppo_monaco", ["stdout", "tensorboard"]))

    print(f"\n[Train] Monaco D42 — 3-stage curriculum: {TOTAL_STEPS:,} steps")
    print(f"        Track: Monaco 2023 Q (~3248 m, 259 waypoints)")
    print(f"        max_steps=6000, multi_lap=True")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,
        callback=curriculum_cb,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std = model.policy.log_std.tolist()

    # Key diagnostic: how much do the curvature columns (dims 5, 6, 7) contribute?
    curv_near_col = model.policy.mlp_extractor.policy_net[0].weight[:, 5].abs().mean().item()
    curv_mid_col  = model.policy.mlp_extractor.policy_net[0].weight[:, 6].abs().mean().item()
    curv_far_col  = model.policy.mlp_extractor.policy_net[0].weight[:, 7].abs().mean().item()

    print(f"\n[Diag] Post-training weights:")
    print(f"       policy_net[0] abs_mean = {pnet0:.6f}")
    print(f"       policy_net[2] abs_mean = {pnet2:.6f}")
    print(f"       col[5] (curv_near) abs_mean = {curv_near_col:.6f}")
    print(f"       col[6] (curv_mid)  abs_mean = {curv_mid_col:.6f}")
    print(f"       col[7] (curv_far)  abs_mean = {curv_far_col:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std)}]")

    save_path = str(project_root / "rl" / "ppo_monaco.zip")
    model.save(save_path)
    print(f"\n[Train] Saved Monaco model to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")


if __name__ == "__main__":
    train()
