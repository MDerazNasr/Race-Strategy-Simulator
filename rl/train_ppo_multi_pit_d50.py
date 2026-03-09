"""
PPO Multi-Agent Pit Strategy D50 — Faster Opponent (opp_max_speed 25 → 28 m/s).

WHY D50?
=========
D48b achieved the best combined pit+racing result in the project:
  - 14 laps, 4 pits at tl≈0.49, 100% completion, ego ahead 64.2%
  - Reward=6122 (+76% vs D37's 3477)

But ego speed (23.53 m/s) is well below D46's 27.28 m/s. Root cause: opponent at
25 m/s is slow enough that the ego can dominate position without running fast.
The pits give fresh-tyre bursts that push it ahead, but between pits it coasts.

FIX: Raise opp_max_speed 25 → 28 m/s.

Why 28 (not 30)?
  D46 pattern: ego ≈ opp + 2.3 m/s when pressured (27.28 vs 25.0).
  With opp=28: target ego ≈ 30 m/s (the car's natural speed ceiling).
  With opp=30: ego may not be able to match → follow equilibrium risk.
  28 is the sweet spot: pressures without exceeding ego's speed ceiling.

With pits:
  - Fresh tyres → max_accel higher → ego can burst to 30 m/s after each pit
  - Worn tyres → ego slows naturally → creates incentive to pit sooner
  - Opponent has no degradation → stays at 28 m/s → ego needs undercut to stay ahead

WARM-START: D48b (ppo_multi_pit_d48b.zip, 14D, 3D, PitAwarePolicy)
  D48b has: pit timing (4 pits at tl≈0.49), positional awareness (col[12]=0.012),
  stable driving (100% completion). Just needs competitive pressure to push speed.
  3M steps should be sufficient — behaviour is stable, only speed needs to increase.

SUCCESS CRITERIA:
  - Completion ≥ 100% (preserve D48b)
  - Pits ≥ 3 (preserve D48b's 4-pit strategy)
  - col[12] (track_gap) weight > 0 (positional awareness preserved)
  - Speed ≥ 25 m/s (improve from D48b's 23.53)
  - Reward ≥ 6122 (at least match D48b)

SAVES TO: rl/ppo_multi_pit_d50.zip
LOGS TO:  runs/ppo_multi_pit_d50/
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

from rl.make_env import make_env_multi_pit_d50
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — must import before PPO.load
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_multi_pit_d48b.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_multi_pit_d48b.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_multi_pit_d48b.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D48b: 6122 reward, 14 laps, 23.53 m/s, 4 pits, 100% completion")
    print(f"        Fix: opp_max_speed 25 → 28 m/s (raise competitive pressure)")

    print(f"\n[Train] Loading D48b checkpoint...")
    model = PPO.load(checkpoint_path, device=device)

    d48b_pnet0  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d48b_tg_col = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()
    d48b_tl_col = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    d48b_pit128 = model.policy.action_net.weight[2, 128].item()
    d48b_b_pit  = model.policy.action_net.bias[2].item()
    d48b_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] D48b loaded weights:")
    print(f"       policy_net[0] abs_mean = {d48b_pnet0:.6f}")
    print(f"       col[11] (tyre_life) abs_mean = {d48b_tl_col:.6f}")
    print(f"       col[12] (track_gap) abs_mean = {d48b_tg_col:.6f}")
    print(f"       action_net[2, 128]   = {d48b_pit128:+.4f}")
    print(f"       action_net.bias[2]   = {d48b_b_pit:+.4f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in d48b_log_std)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in d48b_log_std)}]")

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

    # ── Set environment ────────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_multi_pit_d50] * N_ENVS)
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

    model.set_logger(configure("runs/ppo_multi_pit_d50", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent Pit D50 — Faster Opponent: {TOTAL_STEPS:,} steps")
    print(f"        opp_max_speed: 25 → 28 m/s (forces ego to push harder)")
    print(f"        position_bonus: 1.0 (unchanged from D48b)")
    print(f"        Target: completion=100%, pits≥3, speed≥25 m/s, reward≥6122")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    tg_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()
    tl_col_after  = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_after = model.policy.mlp_extractor.policy_net[0].weight[:, 13].abs().mean().item()
    pit128_after  = model.policy.action_net.weight[2, 128].item()
    b_pit_after   = model.policy.action_net.bias[2].item()
    log_std_after = model.policy.log_std.tolist()

    print(f"\n[Diag] Post-training weights (vs D48b):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [D48b={d48b_pnet0:.6f}]")
    print(f"       col[11] (tyre_life)    = {tl_col_after:.6f}  [D48b={d48b_tl_col:.6f}]")
    print(f"       col[12] (track_gap)    = {tg_col_after:.6f}  [D48b={d48b_tg_col:.6f}]")
    print(f"       col[13] (opp_speed)    = {osp_col_after:.6f}")
    print(f"       action_net[2, 128]     = {pit128_after:+.4f}  [D48b={d48b_pit128:+.4f}]")
    print(f"       action_net.bias[2]     = {b_pit_after:+.4f}  [D48b={d48b_b_pit:+.4f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    threshold_tl = -b_pit_after / pit128_after if pit128_after != 0 else float('nan')
    print(f"\n[Pit] Effective threshold: tl ≈ {threshold_tl:.3f}")

    if tg_col_after > d48b_tg_col:
        print(f"[Strategy] col[12] INCREASED {d48b_tg_col:.6f} → {tg_col_after:.6f} ✓")
    elif tg_col_after > 0.005:
        print(f"[Strategy] col[12] preserved = {tg_col_after:.6f} ✓")
    else:
        print(f"[Strategy] col[12] collapsed = {tg_col_after:.6f} ✗")

    save_path = str(project_root / "rl" / "ppo_multi_pit_d50.zip")
    model.save(save_path)
    print(f"\n[Train] Saved D50 to {save_path}")
    print(f"        Target: completion=100%, pits≥3, speed≥25 m/s, reward≥6122")


if __name__ == "__main__":
    train()
