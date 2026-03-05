"""
PPO Multi-Agent Racing D39 — Ego vs ExpertDriver Opponent (Week 6).

WHY D39? NEW DIRECTION: COMPETITIVE RACING
==========================================
D37 (pit strategy) and cv2 (pure speed) have both plateaued in single-agent
environments:
  - cv2:  4531 reward, 17 laps, 26.92 m/s — no pit, pure speed champ
  - D37:  4377 reward, 15 laps, 23.67 m/s — 3 pits, project best pit policy

The next frontier is competitive racing: a second car on track. The ego PPO
agent must now RACE, not just drive fast in isolation. An ExpertDriver opponent
at 22 m/s provides a meaningful but beatable target.

KEY NEW REWARD TERMS:
  position_bonus   = +0.5/step when ego is ahead (track_gap < 0)
  overtake_bonus   = +200 one-time when ego transitions from behind to ahead
  collision_penalty= -0.5/step within 3m of opponent
  overtake_cooldown= 200 steps (prevents bonus farming from lapping)

OBSERVATION EXTENSION:
  cv2 was trained with 11D obs. We extend to 13D:
    Dim 11: track_gap ∈ [-1, 1] (positive = opponent ahead)
    Dim 12: opp_speed_norm = clip(opp.v / 30.0, 0, 1)

  extend_obs_dim(model, 11, 13) zero-pads the new input columns so the
  policy behaves identically to cv2 on the first step. The agent then
  learns to use track_gap and opp_speed_norm to build race strategy.

  IMPORTANT: extend_obs_dim sets new dim bounds as low=0.0, high=1.0.
  Dim 11 (track_gap) needs low=-1.0. We manually override after the call.

ENTROPY COEFFICIENT:
  D38 showed that ent_coef=0 causes log_std collapse during fine-tuning.
  We set ent_coef=0.01 FROM THE START to prevent this.
  This adds entropy bonus: L = L_clip + c_v×L_value + 0.01×H(π)
  → pushes log_std UP, maintaining exploration throughout training.

TRAINING SETUP:
  Start:  ppo_curriculum_v2.zip (11D → extended to 13D)
  Steps:  3M (more than d36/d37 because the task is harder)
  LR:     cosine 1e-4 → 1e-6
  Env:    F1MultiAgentEnv (ego vs ExpertDriver at max_speed=22 m/s)
  Envs:   4 parallel (DummyVecEnv)
  Freeze: NONE — full unfreeze, all 36,870 params trainable

SUCCESS CRITERIA:
  - Ego spends >50% of time ahead of opponent (avg track_gap < 0) on fixed start
  - Fixed-start reward ≥ 3000 (maintains base driving quality from cv2)
  - At least 1 successful overtake per fixed-start episode on average
  - No catastrophic collapse (laps ≥ 10 on fixed start)

SAVES TO: rl/ppo_multi_agent_d39.zip
LOGS TO:  runs/ppo_multi_agent_d39/
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3.common.buffers import RolloutBuffer

from rl.make_env import make_env_multi_agent
from rl.bc_init_policy import extend_obs_dim
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 3_000_000
    ENT_COEF    = 0.01
    N_ENVS      = 1   # Single env — PPO.load sets n_envs=1, buffer is pre-created

    checkpoint_path = str(project_root / "rl" / "ppo_curriculum_v2.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_curriculum_v2.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_curriculum_v2.py first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        cv2: 4531 reward, 17 laps, 26.92 m/s — best speed policy")

    # ── Load cv2 (11D) ────────────────────────────────────────────────────────
    # Load without env first to inspect weights, then extend obs dim.
    print(f"\n[Train] Loading ppo_curriculum_v2...")
    model = PPO.load(checkpoint_path, device=device)

    # Capture cv2 diagnostics
    cv2_pnet0 = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    cv2_pnet2 = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    cv2_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] cv2 loaded weights:")
    print(f"       policy_net[0] abs_mean = {cv2_pnet0:.6f}")
    print(f"       policy_net[2] abs_mean = {cv2_pnet2:.6f}")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in cv2_log_std)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in cv2_log_std)}]")

    # ── Extend 11D → 13D ──────────────────────────────────────────────────────
    # extend_obs_dim zero-pads columns 11 and 12 in policy_net[0] and value_net[0].
    # New obs space bounds after call: dims 11,12 have low=0.0, high=1.0.
    # Dim 11 (track_gap) needs low=-1.0 — override below.
    print(f"\n[Extend] Extending obs 11D → 13D...")
    extend_obs_dim(model, old_dim=11, new_dim=13)

    # Fix track_gap (dim 11) obs bounds: low must be -1.0, not 0.0
    old_space = model.observation_space
    new_low  = old_space.low.copy()
    new_high = old_space.high.copy()
    new_low[11]  = -1.0   # track_gap ∈ [-1, 1]
    new_high[11] =  1.0   # already 1.0, set explicitly for clarity
    new_high[12] =  1.0   # opp_speed_norm ∈ [0, 1], already 1.0
    fixed_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
    model.observation_space        = fixed_space
    model.policy.observation_space = fixed_space
    print(f"[Extend] Fixed dim 11 (track_gap) bounds: [{new_low[11]:.1f}, {new_high[11]:.1f}]")
    print(f"         Dim 12 (opp_speed_norm) bounds:   [{new_low[12]:.1f}, {new_high[12]:.1f}]")

    # ── Recreate rollout buffer with 13D obs ───────────────────────────────────
    # PPO.load pre-allocates the rollout buffer with the saved obs space (11D).
    # After extend_obs_dim, model.observation_space is 13D, but the buffer arrays
    # are still 11D. We must recreate the buffer before setting the env.
    # This mirrors what SB3 PPO.__init__ does when the model is first created.
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,   # now 13D (after extend_obs_dim + bounds fix)
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=N_ENVS,
    )
    print(f"[Buffer] Rollout buffer recreated: obs_shape={model.observation_space.shape}, n_envs={N_ENVS}")

    # ── Set environment ───────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_multi_agent] * N_ENVS)
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

    model.set_logger(configure("runs/ppo_multi_agent_d39", ["stdout", "tensorboard"]))

    print(f"\n[Train] Multi-Agent D39 — Ego vs ExpertDriver (22 m/s): {TOTAL_STEPS:,} steps")
    print(f"        Starting: cv2 (4531 reward, 17 laps, 26.92 m/s)")
    print(f"        New obs: track_gap (dim 11) + opp_speed_norm (dim 12)")
    print(f"        ent_coef={ENT_COEF} — prevents log_std collapse from d38 lesson")
    print(f"        Target: reward≥3000, laps≥10, ego ahead >50% of time")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    log_std_after = model.policy.log_std.tolist()

    # Track_gap column weights (dim 11 in input)
    tg_col_pnet0 = model.policy.mlp_extractor.policy_net[0].weight[:, 11].abs().mean().item()
    osp_col_pnet0 = model.policy.mlp_extractor.policy_net[0].weight[:, 12].abs().mean().item()

    print(f"\n[Diag] Post-training weights (vs cv2):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [cv2={cv2_pnet0:.6f}]  {(pnet0_after-cv2_pnet0)/cv2_pnet0*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [cv2={cv2_pnet2:.6f}]  {(pnet2_after-cv2_pnet2)/cv2_pnet2*100:+.1f}%")
    print(f"       policy_net[0] col[11] (track_gap)     abs_mean = {tg_col_pnet0:.6f}  [cv2=0.0 init]")
    print(f"       policy_net[0] col[12] (opp_speed_norm) abs_mean = {osp_col_pnet0:.6f}  [cv2=0.0 init]")
    print(f"       log_std = [{', '.join(f'{x:.4f}' for x in log_std_after)}]")
    print(f"       std     = [{', '.join(f'{math.exp(x):.4f}' for x in log_std_after)}]")

    if len(cv2_log_std) == len(log_std_after):
        print(f"       std drift vs cv2: "
              + ", ".join(f"{math.exp(b)-math.exp(a):+.4f}"
                          for a, b in zip(cv2_log_std, log_std_after)))

    save_path = str(project_root / "rl" / "ppo_multi_agent_d39.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d39 to {save_path}")
    print(f"        Run: venv/bin/python rl/evaluate.py")
    print(f"        Target: reward≥3000, ego ahead >50%, ≥1 overtake/episode")


if __name__ == "__main__":
    train()
