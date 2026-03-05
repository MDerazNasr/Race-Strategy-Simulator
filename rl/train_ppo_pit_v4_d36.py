"""
PPO Pit Strategy v4 D36 — Full Unfreeze from D35 (Week 6).

WHY D36?
=========
D32–D35 all use d21's frozen driving features (17.7 m/s), producing 11 laps max.
ppo_curriculum_v2 drives at 26.9 m/s (17 laps). The gap is ~35% speed difference.

FIRST ATTEMPT — CV2 DIRECT TRANSFER (FAILED):
  Tried to copy ppo_curriculum_v2 weights into PitAwarePolicy (11D→12D expansion).
  FAILURE REASON: cv2 was trained WITHOUT tyre degradation. F1Env(tyre_degradation=True)
  reduces mu (friction) linearly with tyre_life: mu = mu_base × max(0.1, tyre_life).
  At tyre_life=0.5: mu drops to half → car understeers/slides → cv2's frozen driving
  policy (optimized for full-grip) crashes immediately (ep_len=100-185 steps, negative reward).

THE D36 FIX — FULL UNFREEZE FROM D35:
  Load d35 (which already works in the tyre degradation env: 11 laps, 2688 reward, 1 pit).
  Unfreeze EVERYTHING: mlp_extractor.policy_net, action_net all rows, log_std.
  Train with LR cosine 1e-4 → 1e-6 (2M steps).

  Why this can improve driving speed:
    - d32-d35 froze features to prevent driving degradation while learning pit row
    - Now that pit row is well-trained (W_pit[128]=-30, good direct connection),
      we can afford to let features evolve
    - With all params trainable, PPO can find a better joint policy: faster driving
      + good pit timing. The voluntary pit reward (+300 at tl<0.60) preserves the
      incentive; W_pit[128]=-30 (very strong signal) is unlikely to be overridden.

  Expected: laps ≥ 12–15, reward > 3000 (vs d35: 11 laps, 2688).

  Risk: pit strategy might degrade. Mitigated by:
    1. Starting from d35 (strong pit policy baseline)
    2. Voluntary pit reward +300 still active → gradient points toward pitting
    3. W_pit[128]=-30 is large — gradient needs many steps to significantly change it

STARTING POINT: ppo_pit_v4_d35.zip
SAVES TO: rl/ppo_pit_v4_d36.zip
LOGS TO:  runs/ppo_pit_v4_d36/
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
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — needed for PPO.load
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d35.zip")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ppo_pit_v4_d35.zip not found at {checkpoint_path}.\n"
            "Run train_ppo_pit_v4_d35.py (d35) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d35: 2688 reward, 11 laps, 1 pit)")
    print(f"[Train] Strategy: FULL UNFREEZE — let PPO improve driving speed AND pit strategy")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit_d30])

    # ── Load d35 checkpoint ───────────────────────────────────────────────────
    print(f"\n[Train] Loading d35 checkpoint...")
    model = PPO.load(checkpoint_path, env=env, device=device)

    # Snapshot d35 values for comparison
    d35_pnet0_w  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    d35_pnet2_w  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    d35_thr_w    = model.policy.action_net.weight[0, :].abs().mean().item()
    d35_str_w    = model.policy.action_net.weight[1, :].abs().mean().item()
    d35_pit_w128 = model.policy.action_net.weight[2, 128].item()
    d35_pit_feat = model.policy.action_net.weight[2, :128].abs().mean().item()
    d35_b_pit    = model.policy.action_net.bias[2].item()
    d35_log_std  = model.policy.log_std.tolist()

    print(f"\n[Diag] D35 loaded weights:")
    print(f"       policy_net[0].weight abs_mean = {d35_pnet0_w:.6f}  (driving features)")
    print(f"       policy_net[2].weight abs_mean = {d35_pnet2_w:.6f}  (driving features)")
    print(f"       action_net.weight[0,:] abs_mean = {d35_thr_w:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {d35_str_w:.6f}  (steer)")
    print(f"       action_net.weight[2,:128] abs_mean = {d35_pit_feat:.6f}  (pit features)")
    print(f"       action_net.weight[2, 128]        = {d35_pit_w128:+.4f}  (direct tyre_life→pit)")
    print(f"       action_net.bias[2]               = {d35_b_pit:+.4f}  (pit threshold)")
    print(f"       log_std = [{d35_log_std[0]:.4f}, {d35_log_std[1]:.4f}, {d35_log_std[2]:.4f}]")

    # ── Set LR schedule ────────────────────────────────────────────────────────
    new_lr_schedule = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr_schedule
    model.lr_schedule   = new_lr_schedule
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # ── FULL UNFREEZE: ensure all parameters are trainable ────────────────────
    # D35 froze mlp_extractor.policy_net via requires_grad=False.
    # Re-enable gradients on all policy parameters.
    unfrozen = 0
    for param in model.policy.parameters():
        param.requires_grad = True
        unfrozen += param.numel()

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Unfreeze] All policy parameters trainable: {trainable:,} / {total:,}")
    print(f"           (vs d35: 131 / {total:,})")
    print(f"           mlp_extractor.policy_net: now TRAINABLE (was frozen in d32-d35)")
    print(f"           action_net all rows: now TRAINABLE (rows 0,1 were frozen in d32-d35)")
    print(f"           NOTE: No gradient hooks — ALL params update freely")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d36", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D36 — Full Unfreeze: {TOTAL_STEPS:,} steps")
    print(f"        Starting from: d35 (2688 reward, 11 laps, 1 pit)")
    print(f"        All {total:,} params trainable — PPO can improve driving AND pit")
    print(f"        W_pit[128] init: {d35_pit_w128:+.4f} (very strong direct signal)")
    print(f"        Voluntary pit reward +300 still active → pit incentive preserved")
    print(f"        Goal: reward>3000, laps≥13, pits≥1 (hopefully 2)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pnet0_after  = model.policy.mlp_extractor.policy_net[0].weight.abs().mean().item()
    pnet2_after  = model.policy.mlp_extractor.policy_net[2].weight.abs().mean().item()
    thr_after    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_after    = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_feat_after = model.policy.action_net.weight[2, :128].abs().mean().item()
    w_tl_after   = model.policy.action_net.weight[2, 128].item()
    b_pit_after  = model.policy.action_net.bias[2].item()
    log_std_after = model.policy.log_std.tolist()

    print(f"\n[Diag] Post-training weights (vs d35):")
    print(f"       policy_net[0] abs_mean = {pnet0_after:.6f}  [d35={d35_pnet0_w:.6f}]  drift={(pnet0_after-d35_pnet0_w)/d35_pnet0_w*100:+.1f}%")
    print(f"       policy_net[2] abs_mean = {pnet2_after:.6f}  [d35={d35_pnet2_w:.6f}]  drift={(pnet2_after-d35_pnet2_w)/d35_pnet2_w*100:+.1f}%")
    print(f"       action_net[0,:] abs_mean = {thr_after:.6f}  [d35={d35_thr_w:.6f}]  (throttle)")
    print(f"       action_net[1,:] abs_mean = {str_after:.6f}  [d35={d35_str_w:.6f}]  (steer)")
    print(f"       action_net[2,:128] abs_mean = {pit_feat_after:.6f}  [d35={d35_pit_feat:.6f}]  (pit features)")
    print(f"       action_net[2, 128]        = {w_tl_after:+.4f}  [d35={d35_pit_w128:+.4f}]  (direct tyre_life)")
    print(f"       action_net.bias[2]        = {b_pit_after:+.4f}  [d35={d35_b_pit:+.4f}]  (pit threshold)")
    print(f"       log_std = [{log_std_after[0]:.4f}, {log_std_after[1]:.4f}, {log_std_after[2]:.4f}]")

    pit_std_after = math.exp(log_std_after[2])
    threshold_tl  = -b_pit_after / w_tl_after if w_tl_after != 0 else float('nan')
    print(f"\n[Diag] Pit signal at key tyre_life values (post-training, direct component):")
    for tl in [0.30, 0.45, 0.53, 0.60, 0.70, 0.80, 0.90]:
        pit_direct = w_tl_after * tl + b_pit_after
        z = -pit_direct / pit_std_after
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        note = "pit" if tl < 0.60 else "hold"
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{note}]")
    print(f"\n[Diag] Effective threshold: tl ≈ {threshold_tl:.3f}")

    save_path = str(project_root / "rl" / "ppo_pit_v4_d36.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d36 to {save_path}")
    print(f"        Goal: faster driving + preserved pit strategy → reward>3000, laps≥13.")


if __name__ == "__main__":
    train()
