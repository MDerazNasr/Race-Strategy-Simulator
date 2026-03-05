"""
PPO Safety Car D40 — Yellow Flag / Safety Car Events (Week 6).

WHY D40? NEW CHALLENGE: RESPOND TO SAFETY CAR PERIODS
=======================================================
D39 (multi-agent racing) revealed a fundamental limitation: the agent
never learned to use opponent-awareness signals (track_gap, opp_speed_norm
carried zero weight). It won purely on raw speed advantage (26.9 vs 22 m/s).

For genuine strategy, the agent needs a situation where speed alone fails.
Safety car periods do exactly this:
  1. Speed limit enforced (22 m/s) — exceeding it costs -9.8 reward/step
  2. Pit stops cheaper under SC (-100 vs -200 net cost) — strategic window
  3. sc_active signal in obs — agent CAN read the flag, must learn to use it

The correct behavior (F1 "undercut"):
  GREEN:  drive at max speed (26+ m/s), pit only when tyres worn
  SC:     slow to 22 m/s AND pit if tyres are >40% worn
         (cheap pit + opponent also slowed = minimal net time loss)

KEY NEW OBSERVATION DIM:
  Dim 12: sc_active ∈ {0.0, 1.0}
    0.0 = green flag (race at full speed)
    1.0 = safety car (slow down, cheap pit window)

SAFETY CAR PARAMETERS:
  sc_trigger_prob = 0.003/step  (~1 SC per 333 steps ≈ 2.5 laps)
  sc_speed_limit  = 22.0 m/s   (same as D39 opponent speed)
  sc_duration     = 80–200 steps (8–20 seconds of SC)
  sc_cooldown     = 300 steps  (minimum gap between SC events)
  sc_speed_penalty = 2.0/m/s   (at cv2 speed: -9.8/step — forces slowdown)
  sc_pit_bonus     = +100      (pit cost -200 + 100 = -100 under SC)

TRAINING SETUP:
  Start:  ppo_pit_v4_d37.zip (12D, 3D action — best pit policy)
  Extend: 12D → 13D (sc_active at dim 12)
  Steps:  3M (same as D39)
  LR:     cosine 1e-4 → 1e-6
  Env:    F1Env(multi_lap=True, tyre_degradation=True, pit_stops=True,
               voluntary_pit_reward=True, safety_car=True)
  Freeze: NONE — full unfreeze, all params trainable
  ent_coef: 0.01 FROM THE START (lesson from d38: prevent log_std collapse)

CRITICAL SB3 PATTERN (same as D39):
  PPO.load without env pre-creates rollout buffer at 12D.
  After extend_obs_dim(model, 12, 13), buffer arrays are still 12D.
  Must recreate RolloutBuffer(13D) before set_env().

SUCCESS CRITERIA:
  - SC compliance: avg speed during SC ≤ 25 m/s
  - SC violations < 30% of SC steps
  - SC pits > outside SC pits (relative rate — undercut learned)
  - col[12] (sc_active) weight in policy_net[0] is non-zero
  - Fixed-start reward ≥ 2500 (below D37 due to SC slowdowns)
  - Laps ≥ 10 on fixed start

SAVES TO: rl/ppo_sc_d40.zip
LOGS TO:  runs/ppo_sc_d40/
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_sc
from rl.bc_init_policy import extend_obs_dim
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1   # PPO.load sets n_envs=1; keep consistent

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d37.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_pit_v4_d37.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_pit_v4_d37.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        d37: 3477 reward, 15 laps, 23.67 m/s, 3 pits — best pit policy")

    # ── Load D37 (12D, 3D action) ──────────────────────────────────────────────
    print(f"\n[Train] Loading ppo_pit_v4_d37...")
    model = PPO.load(checkpoint_path, device=device)

    # Capture D37 diagnostics
    d37_pnet0  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d37_pnet2  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    d37_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] D37 loaded weights:")
    print(f"       policy_net[0] abs_mean = {d37_pnet0:.6f}")
    print(f"       policy_net[2] abs_mean = {d37_pnet2:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in d37_log_std)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in d37_log_std)}]")

    # ── Extend 12D → 13D ──────────────────────────────────────────────────────
    # extend_obs_dim zero-pads columns 12 in policy_net[0] and value_net[0].
    # New obs space bounds: dim 12 has low=0.0, high=1.0 (correct for sc_active).
    # No manual bounds fix needed unlike D39 (track_gap needed low=-1.0).
    print(f"\n[Extend] Extending obs 12D → 13D (sc_active at dim 12)...")
    extend_obs_dim(model, old_dim=12, new_dim=13)
    print(f"[Extend] Dim 12 (sc_active) bounds: [0.0, 1.0] (extend_obs_dim defaults correct)")

    # ── Recreate rollout buffer with 13D obs ───────────────────────────────────
    # PPO.load pre-allocates the rollout buffer with the saved obs space (12D).
    # After extend_obs_dim, model.observation_space is 13D, but the buffer arrays
    # are still 12D. Must recreate the buffer before setting the env.
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,   # now 13D
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=N_ENVS,
    )
    print(f"[Buffer] Rollout buffer recreated: obs_shape={model.observation_space.shape}, n_envs={N_ENVS}")

    # ── Set environment ───────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_sc] * N_ENVS)
    model.set_env(env)

    # ── Learning rate schedule ─────────────────────────────────────────────────
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # ── Entropy coefficient ────────────────────────────────────────────────────
    # Set ent_coef FROM THE START (lesson from d38 — prevents log_std collapse).
    model.ent_coef = ENT_COEF
    print(f"[Entropy] ent_coef = {ENT_COEF} (from the start — prevents std collapse)")

    # ── Full unfreeze ──────────────────────────────────────────────────────────
    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")

    model.set_logger(configure("runs/ppo_sc_d40", ["stdout", "tensorboard"]))

    print(f"\n[Train] Safety Car D40 — SC events + pit undercut: {TOTAL_STEPS:,} steps")
    print(f"        Starting: D37 (3477 reward, 15 laps, 3 pits, 23.67 m/s)")
    print(f"        New obs:  sc_active (dim 12) ∈ {{0, 1}}")
    print(f"        SC:       trigger_prob=0.003, speed_limit=22 m/s, penalty=2.0/m/s")
    print(f"        Pit:      discount -200→-100 under SC (undercut incentive)")
    print(f"        ent_coef={ENT_COEF} — prevents log_std collapse")
    print(f"        Target:   reward≥2500, laps≥10, SC compliance, col[12] non-zero")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()

    # SC column weights (dim 12 in input)
    sc_col_pnet0 = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs d37):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [d37={d37_pnet0:.6f}]  {(pnet0_after-d37_pnet0)/d37_pnet0*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [d37={d37_pnet2:.6f}]  {(pnet2_after-d37_pnet2)/d37_pnet2*100:+.1f}%")
    print(f"       policy_net[0] col[12] (sc_active) abs_mean = {sc_col_pnet0:.6f}  [d37=0.0 init]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    if len(d37_log_std) == len(log_std_after):
        print(f"       std drift vs d37: "
              + ", ".join(f"{math.exp(b)-math.exp(a):+.4f}"
                          for a, b in zip(d37_log_std, log_std_after)))

    if sc_col_pnet0 > 1e-4:
        print(f"\n[Result] col[12] (sc_active) weight = {sc_col_pnet0:.6f} — agent IS using SC signal ✓")
    else:
        print(f"\n[Result] col[12] (sc_active) weight ≈ 0 — agent NOT using SC signal (pure speed strategy)")

    save_path = str(project_root / "rl" / "ppo_sc_d40.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d40 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")
    print(f"        Target: reward≥2500, SC compliance (avg speed≤25 during SC), col[12] non-zero")


if __name__ == "__main__":
    train()
