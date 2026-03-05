"""
PPO Pit Strategy v4 D38 — Entropy-Boosted Training from D37 (Week 6).

WHY D38?
=========
D37 trained 2M more steps from d36 but converged to a local optimum:
  - Steer log_std collapsed: -0.57 → -2.93 (std 0.57 → 0.05, near-deterministic)
  - Throttle log_std: -0.26 → -1.71 (std 0.77 → 0.18)
  - Result: smoother driving, -30% lateral error, but NO increase in laps or speed
  - Same 15-lap/3-pit ceiling as d36

The root cause: PPO with ent_coef=0.0 (default) has no explicit pressure to
maintain exploration. Once the policy finds a good-enough trajectory (15 laps),
log_std collapses and the policy cannot escape the local optimum.

FIX: ent_coef=0.01 adds an entropy bonus to the PPO objective:
  L = L_clip + c_v × L_value − 0.01 × H(π)  [SB3 sign convention: +ent_coef bonus]
The entropy gradient ∂H/∂log_std = +1 for each dimension → pushes log_std UP.
This counteracts the natural collapse, restoring stochastic exploration that can
find faster lines and potentially unlock lap 16.

WHY 0.01?
  - Too small (0.001): likely insufficient to overcome log_std collapse
  - 0.01: standard RL default, ~10% of typical reward magnitude relative weighting
  - Too large (0.1): overwhelms the reward signal, destabilizes driving

RISK: The entropy bonus could push pit log_std too high → noisy pit decisions →
disrupted 3-pit structure. We monitor W_pit[128] and threshold carefully.

Gap to close: curriculum v2 (4531 reward, 17 laps, 26.92 m/s, no pits)
  D37: 3477 reward, 15 laps, 23.67 m/s, 3 pits
  D38 target: reward>3600, laps≥16, speed≥24 m/s, pits≥2

STARTING POINT: ppo_pit_v4_d37.zip
SAVES TO: rl/ppo_pit_v4_d38.zip
LOGS TO:  runs/ppo_pit_v4_d38/
"""

import math
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env_pit_d30
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000
    ENT_COEF    = 0.01

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d37.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_pit_v4_d37.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_pit_v4_d37.py (d37) first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D37: 3477 reward, 15 laps, 23.67 m/s, 3 pits — entropy-boosted unfreeze")

    env = DummyVecEnv([make_env_pit_d30])

    print(f"\n[Train] Loading d37 checkpoint...")
    model = PPO.load(checkpoint_path, env=env, device=device)

    # Capture d37 diagnostics before any changes
    d37_pnet0   = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d37_pnet2   = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    d37_thr     = model.policy.action_net.weight[0, :].abs().mean().item()
    d37_str     = model.policy.action_net.weight[1, :].abs().mean().item()
    d37_pit128  = model.policy.action_net.weight[2, 128].item()
    d37_b_pit   = model.policy.action_net.bias[2].item()
    d37_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] D37 loaded weights:")
    print(f"       policy_net[0] abs_mean = {d37_pnet0:.6f}")
    print(f"       policy_net[2] abs_mean = {d37_pnet2:.6f}")
    print(f"       action_net[0,:] abs_mean = {d37_thr:.6f}  (throttle)")
    print(f"       action_net[1,:] abs_mean = {d37_str:.6f}  (steer)")
    print(f"       action_net[2, 128]        = {d37_pit128:+.4f}  (direct tyre_life)")
    print(f"       action_net.bias[2]        = {d37_b_pit:+.4f}  (pit threshold)")
    print(f"       log_std = [{d37_log_std[0]:.4f}, {d37_log_std[1]:.4f}, {d37_log_std[2]:.4f}]")
    print(f"       std     = [{math.exp(d37_log_std[0]):.4f}, {math.exp(d37_log_std[1]):.4f}, {math.exp(d37_log_std[2]):.4f}]")
    print(f"       Note: steer std={math.exp(d37_log_std[1]):.3f} is collapsed — ent_coef will push it back up")

    # Fresh LR schedule — reset to 1e-4 for full 2M budget
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # Entropy coefficient — THE KEY D38 CHANGE
    model.ent_coef = ENT_COEF
    print(f"[Entropy] ent_coef set to {ENT_COEF} (was {0.0} in d36/d37)")
    print(f"          Entropy gradient pushes log_std UP, counteracting collapse")
    print(f"          Steer std: {math.exp(d37_log_std[1]):.3f} → expected to rise toward ~0.3-0.5")

    # Full unfreeze — all params trainable
    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")

    model.set_logger(configure("runs/ppo_pit_v4_d38", ["stdout", "tensorboard"]))

    print(f"\n[Train] Pit Strategy v4 D38 — Entropy-Boosted: {TOTAL_STEPS:,} steps")
    print(f"        Starting from: d37 (3477 reward, 15 laps, 23.67 m/s, 3 pits)")
    print(f"        ent_coef={ENT_COEF} to restore steer exploration (std: 0.05 → ~0.3)")
    print(f"        Target: reward>3600, laps≥16, speed≥24 m/s, pits≥2")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # Post-training diagnostics
    pnet0_after  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    thr_after    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_after    = model.policy.action_net.weight[1, :].abs().mean().item()
    pit128_after = model.policy.action_net.weight[2, 128].item()
    b_pit_after  = model.policy.action_net.bias[2].item()
    log_std_after = model.policy.log_std.tolist()

    print(f"\n[Diag] Post-training weights (vs d37):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [d37={d37_pnet0:.6f}]  {(pnet0_after-d37_pnet0)/d37_pnet0*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [d37={d37_pnet2:.6f}]  {(pnet2_after-d37_pnet2)/d37_pnet2*100:+.1f}%")
    print(f"       action_net[0,:] abs_mean = {thr_after:.6f}  [d37={d37_thr:.6f}]  (throttle)")
    print(f"       action_net[1,:] abs_mean = {str_after:.6f}  [d37={d37_str:.6f}]  (steer)")
    print(f"       action_net[2, 128]        = {pit128_after:+.4f}  [d37={d37_pit128:+.4f}]")
    print(f"       action_net.bias[2]        = {b_pit_after:+.4f}  [d37={d37_b_pit:+.4f}]")
    print(f"       log_std = [{log_std_after[0]:.4f}, {log_std_after[1]:.4f}, {log_std_after[2]:.4f}]")
    print(f"       std     = [{math.exp(log_std_after[0]):.4f}, {math.exp(log_std_after[1]):.4f}, {math.exp(log_std_after[2]):.4f}]")

    pit_std_after = math.exp(log_std_after[2])
    threshold_tl  = -b_pit_after / pit128_after if pit128_after != 0 else float('nan')
    print(f"\n[Diag] Pit signal at key tyre_life values:")
    for tl in [0.30, 0.45, 0.53, 0.60, 0.70, 0.80, 0.90]:
        pit_direct = pit128_after * tl + b_pit_after
        z = -pit_direct / pit_std_after
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}")
    print(f"[Diag] Effective threshold: tl ≈ {threshold_tl:.3f}")

    save_path = str(project_root / "rl" / "ppo_pit_v4_d38.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d38 to {save_path}")
    print(f"        Target: reward>3600, laps≥16, speed≥24 m/s, pits≥2.")


if __name__ == "__main__":
    train()
