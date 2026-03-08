"""
PPO Multi-Agent Pit Strategy D48b — Stability Fix (position_bonus 2.0 → 1.0).

WHY D48b?
==========
D48 achieved two firsts simultaneously:
  - Pit knowledge transferred: 1 voluntary pit at tyre_life=0.512 ✓
  - Positional awareness emerged: col[12]=0.023 ✓

But crashed every fixed-start episode (0% completion, laps=6 vs D37's 15).
The root cause: position_bonus=2.0 (+4000/episode potential) overweighted
position relative to the crash penalty (−1.0/step × remaining_steps at crash).
The agent learned to race aggressively for position but lost track awareness.

FIX: Halve the position bonus (2.0 → 1.0).

Why 1.0 (not 0.5)?
  D43 tried position_bonus=0.5 → follow equilibrium (agent followed behind,
  no overtaking). D44 raised to 2.0 to break this. D48 showed 2.0 is too much.
  1.0 is the midpoint: +2000/episode for being ahead all episode — meaningful
  incentive but balanced against driving quality reward.

WARM-START: D48 (ppo_multi_pit_d48.zip, 14D, 3D action, PitAwarePolicy)
  D48 already has combined features — pit timing AND positional awareness.
  We just need to re-balance the reward signal, not re-learn from scratch.
  3M steps should be sufficient (vs 5M for D48 from scratch-ish start).

SUCCESS CRITERIA:
  - Completion > 50% (recover from D48's 0%)
  - Pits ≥ 1 (preserved from D48)
  - col[12] (track_gap) weight > 0 (preserved from D48's 0.023)
  - Laps > 6 (improve from D48)
  - Reward ≥ 3477 (at least match D37 baseline)

SAVES TO: rl/ppo_multi_pit_d48b.zip
LOGS TO:  runs/ppo_multi_pit_d48b/
"""

import math
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_multi_pit_d48b
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — must import before PPO.load
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_multi_pit_d48.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_pit_d48.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_pit.py (d48) first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D48: 3524 reward, 6 laps, 23.63 m/s, 1 pit, 76.9% ahead, 0% completion")
    print(f"        Fix: position_bonus 2.0 → 1.0 (restore stability)")

    print(f"\n[Train] Loading D48 checkpoint...")
    model = PPO.load(checkpoint_path, device=device)

    d48_pnet0  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d48_tg_col = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()
    d48_tl_col = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    d48_pit128 = model.policy.action_net.weight[2, 128].item()
    d48_b_pit  = model.policy.action_net.bias[2].item()
    d48_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] D48 loaded weights:")
    print(f"       policy_net[0] abs_mean = {d48_pnet0:.6f}")
    print(f"       col[11] (tyre_life) abs_mean = {d48_tl_col:.6f}")
    print(f"       col[12] (track_gap) abs_mean = {d48_tg_col:.6f}")
    print(f"       action_net[2, 128]   = {d48_pit128:+.4f}")
    print(f"       action_net.bias[2]   = {d48_b_pit:+.4f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in d48_log_std)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in d48_log_std)}]")

    # ── Recreate rollout buffer ────────────────────────────────────────────────
    # PPO.load pre-allocates the buffer with saved obs/action shapes. Recreate
    # to ensure clean state before changing env.
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

    # ── Set environment ────────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_multi_pit_d48b] * N_ENVS)
    model.set_env(env)

    # ── Fresh LR schedule ──────────────────────────────────────────────────────
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # ── Reset Adam optimizer ───────────────────────────────────────────────────
    import torch.optim as optim
    model.policy.optimizer = optim.Adam(
        model.policy.parameters(), lr=1e-4, eps=1e-5
    )
    print(f"[Adam] Optimizer reset — fresh momentum")

    # ── Entropy coefficient ────────────────────────────────────────────────────
    model.ent_coef = ENT_COEF
    print(f"[Entropy] ent_coef = {ENT_COEF}")

    # ── Full unfreeze ──────────────────────────────────────────────────────────
    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,}")

    model.set_logger(configure("runs/ppo_multi_pit_d48b", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent Pit D48b — Stability Fix: {TOTAL_STEPS:,} steps")
    print(f"        position_bonus: 2.0 → 1.0 (balance position vs stability)")
    print(f"        Target: completion>50%, pits≥1, col[12]>0, laps>6")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    tg_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()
    tl_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 13].abs().mean().item()
    pit128_after = model.policy.action_net.weight[2, 128].item()
    b_pit_after  = model.policy.action_net.bias[2].item()
    log_std_after = model.policy.log_std.tolist()

    print(f"\n[Diag] Post-training weights (vs D48):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [D48={d48_pnet0:.6f}]")
    print(f"       col[11] (tyre_life)    = {tl_col_after:.6f}  [D48={d48_tl_col:.6f}]")
    print(f"       col[12] (track_gap)    = {tg_col_after:.6f}  [D48={d48_tg_col:.6f}]")
    print(f"       col[13] (opp_speed)    = {osp_col_after:.6f}")
    print(f"       action_net[2, 128]     = {pit128_after:+.4f}  [D48={d48_pit128:+.4f}]")
    print(f"       action_net.bias[2]     = {b_pit_after:+.4f}  [D48={d48_b_pit:+.4f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    pit_std_after = math.exp(log_std_after[2])
    threshold_tl  = -b_pit_after / pit128_after if pit128_after != 0 else float('nan')
    print(f"\n[Pit] Effective threshold: tl ≈ {threshold_tl:.3f}")

    if tg_col_after > 0.01:
        print(f"[Strategy] col[12] weight = {tg_col_after:.6f} > 0 → positional awareness PRESERVED ✓")
    else:
        print(f"[Strategy] col[12] weight = {tg_col_after:.6f} ≈ 0 → positional strategy may have collapsed")

    save_path = str(project_root / "rl" / "ppo_multi_pit_d48b.zip")
    model.save(save_path)
    print(f"\n[Train] Saved D48b to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")
    print(f"        Target: completion>50%, pits≥1, col[12]>0, reward>3477")


if __name__ == "__main__":
    train()
