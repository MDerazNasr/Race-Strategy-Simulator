"""
PPO Pit Strategy v4 D37 — Continue Full Unfreeze from D36 (Week 6).

WHY D37?
=========
D36 (full unfreeze from d35) achieved 3427 reward, 15 laps, 23.60 m/s, 3 pits.
Training ep_rew_mean was still at ~2250 at the final iteration — the policy had
not fully converged. Training ep_len ≈ 1200 (vs 2000 max), meaning ~40% of
random-start training episodes still crash. More training should:
  1. Improve driving speed further (23.60 → closer to cv2's 26.92 m/s)
  2. Improve robustness (fewer crashes in random-start training)
  3. Possibly reach 16–17 laps in fixed-start evaluation

Gap to close: curriculum v2 (4531 reward, 17 laps, 26.92 m/s, no pits)
  D36: 3427 reward, 15 laps, 23.60 m/s, 3 pits
  D37 target: reward>3700, laps≥16, speed>25 m/s, pits≥2

APPROACH:
  Load d36, unfreeze all params, LR cosine 1e-4 → 1e-6 (fresh LR schedule),
  train 2M more steps. Same setup as d36 — just more budget.

STARTING POINT: ppo_pit_v4_d36.zip
SAVES TO: rl/ppo_pit_v4_d37.zip
LOGS TO:  runs/ppo_pit_v4_d37/
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

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d36.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_pit_v4_d36.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_pit_v4_d36.py (d36) first."
        )
    print(f"[Train] Starting from: {checkpoint_path}")
    print(f"        D36: 3427 reward, 15 laps, 23.60 m/s, 3 pits — continuing full unfreeze")

    env = DummyVecEnv([make_env_pit_d30])

    print(f"\n[Train] Loading d36 checkpoint...")
    model = PPO.load(checkpoint_path, env=env, device=device)

    d36_pnet0 = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d36_pnet2 = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    d36_thr   = model.policy.action_net.weight[0, :].abs().mean().item()
    d36_str   = model.policy.action_net.weight[1, :].abs().mean().item()
    d36_pit128= model.policy.action_net.weight[2, 128].item()
    d36_b_pit = model.policy.action_net.bias[2].item()
    d36_log_std = model.policy.log_std.tolist()

    print(f"\n[Diag] D36 loaded weights:")
    print(f"       policy_net[0] abs_mean = {d36_pnet0:.6f}")
    print(f"       policy_net[2] abs_mean = {d36_pnet2:.6f}")
    print(f"       action_net[0,:] abs_mean = {d36_thr:.6f}  (throttle)")
    print(f"       action_net[1,:] abs_mean = {d36_str:.6f}  (steer)")
    print(f"       action_net[2, 128]        = {d36_pit128:+.4f}  (direct tyre_life)")
    print(f"       action_net.bias[2]        = {d36_b_pit:+.4f}  (pit threshold)")
    print(f"       log_std = [{d36_log_std[0]:.4f}, {d36_log_std[1]:.4f}, {d36_log_std[2]:.4f}]")

    # Fresh LR schedule — reset to 1e-4 for full 2M budget
    new_lr = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr
    model.lr_schedule   = new_lr
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # Ensure all params trainable (d36 was already fully unfrozen, but re-confirm)
    for param in model.policy.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")

    model.set_logger(configure("runs/ppo_pit_v4_d37", ["stdout", "tensorboard"]))

    print(f"\n[Train] Pit Strategy v4 D37 — Continue Full Unfreeze: {TOTAL_STEPS:,} steps")
    print(f"        Starting from: d36 (3427 reward, 15 laps, 23.60 m/s, 3 pits)")
    print(f"        Target: reward>3700, laps≥16, speed>25 m/s")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # Post-training diagnostics
    pnet0_after = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    thr_after   = model.policy.action_net.weight[0, :].abs().mean().item()
    str_after   = model.policy.action_net.weight[1, :].abs().mean().item()
    pit128_after= model.policy.action_net.weight[2, 128].item()
    b_pit_after = model.policy.action_net.bias[2].item()
    log_std_after = model.policy.log_std.tolist()

    print(f"\n[Diag] Post-training weights (vs d36):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [d36={d36_pnet0:.6f}]  {(pnet0_after-d36_pnet0)/d36_pnet0*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [d36={d36_pnet2:.6f}]  {(pnet2_after-d36_pnet2)/d36_pnet2*100:+.1f}%")
    print(f"       action_net[0,:] abs_mean = {thr_after:.6f}  [d36={d36_thr:.6f}]  (throttle)")
    print(f"       action_net[1,:] abs_mean = {str_after:.6f}  [d36={d36_str:.6f}]  (steer)")
    print(f"       action_net[2, 128]        = {pit128_after:+.4f}  [d36={d36_pit128:+.4f}]")
    print(f"       action_net.bias[2]        = {b_pit_after:+.4f}  [d36={d36_b_pit:+.4f}]")
    print(f"       log_std = [{log_std_after[0]:.4f}, {log_std_after[1]:.4f}, {log_std_after[2]:.4f}]")

    pit_std_after = math.exp(log_std_after[2])
    threshold_tl  = -b_pit_after / pit128_after if pit128_after != 0 else float('nan')
    print(f"\n[Diag] Pit signal at key tyre_life values:")
    for tl in [0.30, 0.45, 0.53, 0.60, 0.70, 0.80, 0.90]:
        pit_direct = pit128_after * tl + b_pit_after
        z = -pit_direct / pit_std_after
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}")
    print(f"[Diag] Effective threshold: tl ≈ {threshold_tl:.3f}")

    save_path = str(project_root / "rl" / "ppo_pit_v4_d37.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d37 to {save_path}")
    print(f"        Target: reward>3700, laps≥16, speed>25 m/s, pits≥2.")


if __name__ == "__main__":
    train()
