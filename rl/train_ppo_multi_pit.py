"""
PPO Multi-Agent Pit Strategy D48 — Pit Stops + Racing vs Opponent (Week 7).

WHY D48? COMBINING TWO HARD SKILLS
====================================
Every previous experiment specialised in ONE advanced behaviour:
  - D37: best pit policy (3 voluntary pits, 23.67 m/s, 3477 reward)
  - D46: best positional strategy (7943 reward, track_gap weight 0.000 → 0.070)

D48 is the first experiment to require BOTH simultaneously:
  - Ego must pit at the right time (tyre_life < 0.60)
  - Ego must maintain/take position against a 25 m/s opponent

Key open question: does an UNDERCUT emerge?
  Undercut = pit one stint earlier than "optimal" single-agent timing
  to exit ahead of the opponent (who is still on worn tyres).
  This requires the agent to reason about BOTH obs[11] (own tyre_life)
  AND obs[12] (track_gap) simultaneously — the hardest strategic inference
  in the project.

OBSERVATION EXTENSION (12D → 14D):
  D37 obs: [0-10 standard, 11=tyre_life]
  D48 obs: [0-10 standard, 11=tyre_life, 12=track_gap, 13=opp_speed_norm]

  CRITICAL: tyre_life stays at obs[11] — same index as all D32–D37 experiments.
  PitAwarePolicy.TYRE_LIFE_OBS_IDX = 11, so the direct tyre_life → pit_signal
  connection (the D32 fix) works without any modification.

  Two new columns (dims 12–13) are zero-initialized via extend_obs_dim().
  These start silent and learn through PPO gradient updates.

WARM-START RATIONALE:
  D37 (pit policy) is chosen over D46 (positional policy) because:
  - Pit behavior took 20 experiments (D18–D37) to learn; re-learning from scratch
    would likely fail in 5M steps.
  - D46 (2D action, no pit) cannot directly provide pit_signal knowledge.
  - D37's PitAwarePolicy has the direct tyre_life connection — the architectural
    fix for pit timing. Preserving this is more important than position strategy.
  - Positional strategy (from D37's 0.000 track_gap weight) will re-learn against
    the 25 m/s opponent, guided by position_bonus=2.0 (the D44 fix).

TRAINING SETUP:
  Start:  ppo_pit_v4_d37.zip (12D → extended to 14D)
  Steps:  5M (more budget — two skills to combine from scratch)
  LR:     cosine 1e-4 → 1e-6
  Env:    F1MultiAgentPitEnv (14D, 3D action)
  ent_coef: 0.01 (prevents log_std collapse — d38 lesson)

SUCCESS CRITERIA:
  - Pits ≥ 1 time per fixed-start episode (D37 warm-start should preserve this)
  - col[12] (track_gap) weight > 0 → positional awareness learned
  - Reward > D37's 3477 (combined racing + pit strategy should beat pure pit)
  - Ideally: timing correlation between pit and track_gap position → undercut

SAVES TO: rl/ppo_multi_pit_d48.zip
LOGS TO:  runs/ppo_multi_pit_d48/
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_multi_pit
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — must import before PPO.load
from rl.bc_init_policy import extend_obs_dim
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 5_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d37.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_pit_v4_d37.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_pit_v4_d37.py (d37) first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D37: 3477 reward, 15 laps, 23.67 m/s, 3 pits — best pit policy")

    # ── Load D37 (12D obs, 3D action, PitAwarePolicy) ─────────────────────────
    # PitAwarePolicy must be imported before PPO.load (pickle deserialization).
    print(f"\n[Train] Loading D37 checkpoint...")
    model = PPO.load(checkpoint_path, device=device)

    d37_pnet0   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d37_pnet2   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    d37_pit128  = model.policy.action_net.weight[2, 128].item()
    d37_b_pit   = model.policy.action_net.bias[2].item()
    d37_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] D37 loaded weights:")
    print(f"       policy_net[0] abs_mean = {d37_pnet0:.6f}")
    print(f"       policy_net[2] abs_mean = {d37_pnet2:.6f}")
    print(f"       action_net[2, 128]     = {d37_pit128:+.4f}  (direct tyre_life connection)")
    print(f"       action_net.bias[2]     = {d37_b_pit:+.4f}  (pit threshold)")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in d37_log_std)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in d37_log_std)}]")

    # ── Extend 12D → 14D ──────────────────────────────────────────────────────
    # extend_obs_dim zero-pads columns 12 and 13 in policy_net[0] and value_net[0].
    # Dims 0–11 (including tyre_life at dim 11) are preserved exactly.
    # PitAwarePolicy still reads obs[:, 11] for tyre_life — no changes needed.
    print(f"\n[Extend] Extending obs 12D → 14D...")
    extend_obs_dim(model, old_dim=12, new_dim=14)

    # Fix track_gap (dim 12) obs bounds: extend_obs_dim sets low=0, but track_gap ∈ [-1,1]
    old_space = model.observation_space
    new_low   = old_space.low.copy()
    new_high  = old_space.high.copy()
    new_low[12]  = -1.0   # track_gap ∈ [-1, 1]
    new_high[12] =  1.0
    new_high[13] =  1.0   # opp_speed_norm ∈ [0, 1]
    new_low[11]  =  0.0   # tyre_life ∈ [0, 1] (was already 0, confirm)
    new_high[11] =  1.0   # tyre_life already 1.0 from extend_obs_dim
    fixed_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
    model.observation_space        = fixed_space
    model.policy.observation_space = fixed_space
    print(f"[Extend] Fixed dim 11 (tyre_life) bounds:   [{new_low[11]:.1f}, {new_high[11]:.1f}]")
    print(f"         Fixed dim 12 (track_gap) bounds:   [{new_low[12]:.1f}, {new_high[12]:.1f}]")
    print(f"         Dim 13 (opp_speed_norm) bounds:    [{new_low[13]:.1f}, {new_high[13]:.1f}]")

    # ── Recreate rollout buffer with 14D obs ───────────────────────────────────
    # PPO.load pre-allocates the rollout buffer with saved 12D shape.
    # Must recreate before set_env() so buffer arrays match the new obs size.
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,   # now 14D
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=N_ENVS,
    )
    print(f"[Buffer] Rollout buffer recreated: obs_shape={model.observation_space.shape}, n_envs={N_ENVS}")

    # ── Set environment ────────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_multi_pit] * N_ENVS)
    model.set_env(env)

    # ── Fresh LR schedule ──────────────────────────────────────────────────────
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # ── Reset Adam optimizer ───────────────────────────────────────────────────
    # D37's optimizer has stale momentum from pit-only training.
    # Fresh optimizer prevents interference when learning positional strategy.
    import torch.optim as optim
    model.policy.optimizer = optim.Adam(
        model.policy.parameters(), lr=1e-4, eps=1e-5
    )
    print(f"[Adam] Optimizer reset — fresh momentum for combined pit+position task")

    # ── Entropy coefficient ────────────────────────────────────────────────────
    model.ent_coef = ENT_COEF
    print(f"[Entropy] ent_coef = {ENT_COEF} (from start — prevents log_std collapse)")

    # ── Full unfreeze ──────────────────────────────────────────────────────────
    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")

    # ── Pit signal sanity check at init ────────────────────────────────────────
    pit128_init = model.policy.action_net.weight[2, 128].item()
    b_pit_init  = model.policy.action_net.bias[2].item()
    pit_std_init = math.exp(d37_log_std[2])
    print(f"\n[Pit Init] Direct tyre_life weight: action_net[2, 128] = {pit128_init:+.4f}")
    print(f"[Pit Init] Pit bias: action_net.bias[2] = {b_pit_init:+.4f}")
    print(f"[Pit Init] Pit signal at key tyre_life values (features noise excluded):")
    for tl in [0.30, 0.45, 0.60, 0.70, 0.85, 1.0]:
        pit_direct = pit128_init * tl + b_pit_init
        z = -pit_direct / pit_std_init
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        label = "PIT" if tl < 0.60 else "hold"
        print(f"         tl={tl:.2f}: pit≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{label}]")

    model.set_logger(configure("runs/ppo_multi_pit_d48", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent Pit D48: {TOTAL_STEPS:,} steps")
    print(f"        Starting from: D37 (3477 reward, 15 laps, 23.67 m/s, 3 pits)")
    print(f"        Obs: 12D → 14D (added track_gap dim 12, opp_speed_norm dim 13)")
    print(f"        Env: F1MultiAgentPitEnv (opp=25 m/s, position_bonus=2.0)")
    print(f"        ent_coef={ENT_COEF} — prevents log_std collapse")
    print(f"        Target: pits≥1, col[12]>0, reward>D37's 3477")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()
    pit128_after  = model.policy.action_net.weight[2, 128].item()
    b_pit_after   = model.policy.action_net.bias[2].item()

    # Column weights: tyre_life (11), track_gap (12), opp_speed_norm (13)
    tl_col   = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    tg_col   = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()
    osp_col  = model.policy.mlp_extractor.policy_net[0].weight[:, 13].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs D37):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [D37={d37_pnet0:.6f}]  {(pnet0_after-d37_pnet0)/d37_pnet0*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [D37={d37_pnet2:.6f}]  {(pnet2_after-d37_pnet2)/d37_pnet2*100:+.1f}%")
    print(f"       col[11] (tyre_life)    abs_mean = {tl_col:.6f}  [D37=init]")
    print(f"       col[12] (track_gap)    abs_mean = {tg_col:.6f}  [D37=0.0 init]")
    print(f"       col[13] (opp_speed)    abs_mean = {osp_col:.6f}  [D37=0.0 init]")
    print(f"       action_net[2, 128]     = {pit128_after:+.4f}  [D37={d37_pit128:+.4f}]  (tyre_life direct)")
    print(f"       action_net.bias[2]     = {b_pit_after:+.4f}  [D37={d37_b_pit:+.4f}]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    # Pit threshold at end of training
    pit_std_after = math.exp(log_std_after[2])
    threshold_tl  = -b_pit_after / pit128_after if pit128_after != 0 else float('nan')
    print(f"\n[Pit Final] Effective threshold: tl ≈ {threshold_tl:.3f}")
    print(f"[Pit Final] Pit signal at key tyre_life values:")
    for tl in [0.30, 0.45, 0.60, 0.70, 0.85, 1.0]:
        pit_direct = pit128_after * tl + b_pit_after
        z = -pit_direct / pit_std_after
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        label = "PIT" if tl < 0.60 else "hold"
        print(f"           tl={tl:.2f}: pit≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{label}]")

    if tg_col > 0.01:
        print(f"\n[Strategy] col[12] (track_gap) weight = {tg_col:.6f} > 0 → positional awareness LEARNED ✓")
    else:
        print(f"\n[Strategy] col[12] (track_gap) weight = {tg_col:.6f} ≈ 0 → positional strategy NOT learned")
        print(f"           Consider: raise position_bonus further, or train longer")

    save_path = str(project_root / "rl" / "ppo_multi_pit_d48.zip")
    model.save(save_path)
    print(f"\n[Train] Saved D48 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")
    print(f"        Target: pits≥1, col[12]>0, reward>3477")


if __name__ == "__main__":
    train()
