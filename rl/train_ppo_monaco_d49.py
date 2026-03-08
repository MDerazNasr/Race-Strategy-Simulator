"""
PPO Monaco D49 — No Curriculum, Fixed max_accel=8 m/s².

WHY D49? CURRICULUM IS THE BLOCKER FOR MONACO
===============================================
D42 and D47 both got stuck in Stage 0 (0 laps after 3M steps each).
The STAGES curriculum (designed for the 314m oval) requires 50% lap completion
to advance. Monaco's lap is ~3750m — 12× longer. At curriculum speeds, the agent
covers only ~300m before crashing: 8% of a lap, nowhere near the 50% threshold.

The data collection and BC training ARE working (D47: 24,364 samples, 3 full laps,
BC loss 0.003). The BC-initialised policy knows how to steer around Monaco.
The curriculum is getting in the way by never letting the agent practice fast driving.

D49 fix: skip the curriculum entirely.
  - Set car.max_accel = 8.0 m/s² at init and leave it constant throughout training.
  - max_accel=8 gives top speed ~20 m/s over a 10s run from rest — fast enough
    to be useful on Monaco's straights, controlled enough for hairpins.
  - max_steps=8000 (≈5.5 laps at 20 m/s) — more room to complete laps vs 6000.
  - No CurriculumCallback, no stage logic.

WHY max_accel=8 (not 6 or 15)?
  - 6 m/s² cap (BC expert speed): too slow, no incentive to go faster
  - 15 m/s²  (Stage 3 curriculum): too fast for Monaco hairpins at early training
  - 8 m/s²: at 20 m/s average, Monaco lap ≈ 3750/20 = 188s = 1875 steps.
    Completing 4 laps needs 7500 steps — within max_steps=8000.

WARM-START: BC only (bc_policy_monaco.pt)
  Same initialization as D42/D47. No change here — the BC weights are correct,
  the curriculum was the problem.

SUCCESS CRITERIA:
  - ≥ 1 full Monaco lap completed in evaluation
  - Curvature weights non-zero (confirmed in D42/D47 even in failed policies)
  - ep_len_mean > 2000 (surviving more than half-lap territory)

SAVES TO: rl/ppo_monaco_d49.zip
LOGS TO:  runs/ppo_monaco_d49/
"""

import sys
import math
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_monaco_d49
from rl.bc_init_policy import load_bc_weights_into_ppo
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 5_000_000
    MAX_ACCEL   = 8.0
    MAX_STEPS   = 8000

    bc_path = str(project_root / "bc" / "bc_policy_monaco.pt")
    if not Path(bc_path).exists():
        raise FileNotFoundError(
            f"bc_policy_monaco.pt not found at {bc_path}.\n"
            "Run generate_monaco_dataset() then bc/train_bc_monaco.py first.\n"
            "(D47 already generated this — check bc/ directory.)"
        )
    print(f"[Train] Starting D49 — Monaco PPO, no curriculum")
    print(f"        BC weights: {bc_path}")
    print(f"        max_accel = {MAX_ACCEL} m/s² (fixed, no curriculum)")
    print(f"        max_steps = {MAX_STEPS} per episode")

    # ── Create env (max_accel set inside make_env_monaco_d49) ─────────────────
    vec_env = DummyVecEnv([lambda: make_env_monaco_d49(
        max_steps=MAX_STEPS, max_accel=MAX_ACCEL
    )])

    # ── Build PPO from scratch with BC warm-start ──────────────────────────────
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=cosine_schedule(initial_lr=1e-4, min_lr=1e-6),
        ent_coef=0.005,
        clip_range=0.1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device=device,
    )

    load_bc_weights_into_ppo(model, bc_path, device=device)
    print(f"[BC Init] Loaded BC weights from {bc_path}")

    # Confirm max_accel on the inner env
    inner_env = vec_env.envs[0].env   # unwrap Monitor → F1Env
    print(f"[Env] Inner env max_accel = {inner_env.car.max_accel}")

    # Pre-training diagnostics
    pnet0_init = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    log_std_init = model.policy.log_std.tolist()
    curv_near = model.policy.mlp_extractor.policy_net[0].weight[:, 5].abs().mean().item()
    curv_mid  = model.policy.mlp_extractor.policy_net[0].weight[:, 6].abs().mean().item()
    curv_far  = model.policy.mlp_extractor.policy_net[0].weight[:, 7].abs().mean().item()

    print(f"\n[Diag] BC-init weights:")
    print(f"       policy_net[0] abs_mean = {pnet0_init:.6f}")
    print(f"       col[5] (curv_near) = {curv_near:.6f}")
    print(f"       col[6] (curv_mid)  = {curv_mid:.6f}")
    print(f"       col[7] (curv_far)  = {curv_far:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_init)}]")

    model.set_logger(configure("runs/ppo_monaco_d49", ["stdout", "tensorboard"]))

    print(f"\n[Train] Monaco D49 — No Curriculum: {TOTAL_STEPS:,} steps")
    print(f"        max_accel={MAX_ACCEL} fixed (vs Stage 0=6, Stage 2=11, Stage 3=15)")
    print(f"        max_steps={MAX_STEPS} (vs D42/D47: 6000)")
    print(f"        NO CurriculumCallback — plain PPO from BC init")
    print(f"        Target: ≥1 Monaco lap, curvature weights non-zero")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    # No callback — plain PPO
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()

    curv_near_after = model.policy.mlp_extractor.policy_net[0].weight[:, 5].abs().mean().item()
    curv_mid_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 6].abs().mean().item()
    curv_far_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 7].abs().mean().item()

    print(f"\n[Diag] Post-training weights:")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [init={pnet0_init:.6f}]  {(pnet0_after-pnet0_init)/pnet0_init*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}")
    print(f"       col[5] (curv_near) = {curv_near_after:.6f}  [init={curv_near:.6f}]")
    print(f"       col[6] (curv_mid)  = {curv_mid_after:.6f}  [init={curv_mid:.6f}]")
    print(f"       col[7] (curv_far)  = {curv_far_after:.6f}  [init={curv_far:.6f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    save_path = str(project_root / "rl" / "ppo_monaco_d49.zip")
    model.save(save_path)
    print(f"\n[Train] Saved D49 to {save_path}")
    print(f"        Target: ≥1 Monaco lap, curvature weights non-zero")


if __name__ == "__main__":
    train()
