"""
PPO Pit Strategy v4 D35 — Seal the Second Pit (Week 5/6).

WHY D35?
=========
D34 achieved 1.85 pits average in random-start evaluation (85% completion),
proving the two-pit strategy is learned. But fixed-start evaluation still shows
pits=1.0 — the second pit fires stochastically but not deterministically.

ROOT CAUSE (from d33/d34 analysis):
  At the second-pit window (step ~1314, tl≈0.53 in fixed-start trajectory):
    pit_signal_total = features_contribution + W_pit[128] × tl + b_pit

  D34 post-training values:
    W_pit[128] = -19.97,  b_pit = +12.07
    direct_component = -19.97 × 0.53 + 12.07 = +1.48
    features_contribution ≈ -1.5  (frozen features at that car state)
    net pit_signal ≈ -0.02  → JUST barely negative → no pit in deterministic eval

  The second pit fires stochastically (Gaussian noise pushes it positive most of the
  time), but the deterministic mean is -0.02 — a hair below zero.

THE D35 FIX — STRONGER DIRECT WEIGHT:
  Load d34 weights (W_pit[:128] already reduced -13.5% from d34 training),
  then RE-INITIALIZE with even larger direct weight:

    W_pit[128] = -30.0  (was -20.0 in d34)
    b_pit      = +18.0  (was +12.0 in d34)

  Threshold preserved: -30 × 0.60 + 18 = 0  → pit threshold still tl=0.60 ✓
  Second-pit window:   -30 × 0.53 + 18 = +2.1  → easily overcomes -1.5 features ✓
    net pit_signal ≈ 2.1 + (-1.5) = +0.6  → POSITIVE → second pit fires deterministically!

  D34 already reduced feature weights (0.1998 → 0.1730, −13.5%). Starting from
  d34 means the features contribution at the second-pit window is already smaller
  than in d32/d33, giving the stronger direct weight an even better chance.

Post-init pit probabilities (direct component, std≈0.81 from d34):
  tl=0.30: pit_direct = -30×0.30+18 = +9.0   P(pit>0) ≈ 100%  [always pit]
  tl=0.45: pit_direct = -30×0.45+18 = +4.5   P(pit>0) ≈ 100%  [always pit]
  tl=0.53: pit_direct = -30×0.53+18 = +2.1   P(pit>0) ≈ 100%  [2nd pit window ✓]
  tl=0.60: pit_direct = -30×0.60+18 =  0.0   P(pit>0) ≈  50%  [threshold]
  tl=0.70: pit_direct = -30×0.70+18 = -3.0   P(pit>0) ≈   0%  [hold]
  tl=0.90: pit_direct = -30×0.90+18 = -9.0   P(pit>0) ≈   0%  [never pit]

STARTING POINT: ppo_pit_v4_d34.zip
  - d34 has W_pit[:128] already reduced (features less dominant on pit row)
  - Override: W_pit[128]=-30.0, b_pit=+18.0

SAVES TO: rl/ppo_pit_v4_d35.zip
LOGS TO:  runs/ppo_pit_v4_d35/
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
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — needed for PPO.load deserialization
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d34.zip")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4_d34.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4_d34.py (d34) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d34, 85% random completion, 1.85 pits)")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit_d30])

    # ── Load d34 checkpoint ───────────────────────────────────────────────────
    print(f"\n[Train] Loading d34 checkpoint...")
    model = PPO.load(checkpoint_path, env=env, device=device)

    d34_w_tl       = model.policy.action_net.weight[2, 128].item()
    d34_b_pit      = model.policy.action_net.bias[2].item()
    d34_log_std    = model.policy.log_std[2].item()
    d34_pit_w_mean = model.policy.action_net.weight[2, :128].abs().mean().item()
    d34_thr_w_mean = model.policy.action_net.weight[0, :].abs().mean().item()
    d34_str_w_mean = model.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] D34 loaded weights (before re-initialization):")
    print(f"       action_net.weight[0,:] abs_mean = {d34_thr_w_mean:.6f}  (throttle — frozen)")
    print(f"       action_net.weight[1,:] abs_mean = {d34_str_w_mean:.6f}  (steer — frozen)")
    print(f"       action_net.weight[2,:128] abs_mean = {d34_pit_w_mean:.6f}  (pit features, reduced by d34)")
    print(f"       action_net.weight[2, 128]        = {d34_w_tl:+.4f}   (will be overridden to -30.0)")
    print(f"       action_net.bias[2]               = {d34_b_pit:+.4f}   (will be overridden to +18.0)")
    print(f"       log_std[2]                       = {d34_log_std:.4f}   (pit std = {torch.exp(torch.tensor(d34_log_std)).item():.3f}, kept)")

    assert model.policy.action_net.in_features == 129, "Expected 129-dim action_net"
    print(f"\n[Build] Architecture verified: action_net is Linear(129, 3) ✓")

    # ── RE-INITIALIZE pit row with stronger direct weight ─────────────────────
    # W_pit[128] = -30.0  (was -19.97 in d34)
    # b_pit      = +18.0  (was +12.07 in d34)
    #
    # Threshold: -30 × 0.60 + 18 = 0              → threshold preserved at tl=0.60 ✓
    # 2nd pit:   -30 × 0.53 + 18 = +2.1           → overcomes features_noise ≈ -1.5 ✓
    #            net ≈ 2.1 + (-1.5) = +0.6 → POSITIVE → 2nd pit fires deterministically!
    with torch.no_grad():
        model.policy.action_net.weight[2, 128] = -30.0
        model.policy.action_net.bias[2]        = +18.0

    new_w_tl  = model.policy.action_net.weight[2, 128].item()
    new_b_pit = model.policy.action_net.bias[2].item()
    print(f"\n[Init] Re-initialized pit direct connection:")
    print(f"       W_pit[128]: {d34_w_tl:+.4f} → {new_w_tl:+.4f}  (3× larger magnitude vs d32)")
    print(f"       b_pit:      {d34_b_pit:+.4f} → {new_b_pit:+.4f}  (threshold preserved at tl≈0.60)")

    pit_std = math.exp(d34_log_std)
    print(f"\n[Init] Post-init pit probabilities (direct component, std={pit_std:.3f}):")
    for tl in [0.30, 0.45, 0.53, 0.60, 0.70, 0.80, 0.90]:
        pit_direct = new_w_tl * tl + new_b_pit
        z = -pit_direct / pit_std
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        note = "2nd pit window" if tl == 0.53 else ("pit" if tl < 0.60 else "hold")
        check = "✓" if (tl <= 0.53 and p_pit > 0.95) or (tl >= 0.70 and p_pit < 0.05) else "~"
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{note}] {check}")

    # ── Set LR schedule (same as d32/d34) ─────────────────────────────────────
    new_lr_schedule = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr_schedule
    model.lr_schedule   = new_lr_schedule
    print(f"\n[LR] cosine(1e-4 → 1e-6), start={model.lr_schedule(1.0):.2e}")

    # ── Re-apply three-layer freeze ───────────────────────────────────────────
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    def _hook_weight(grad):
        g = grad.clone(); g[0, :] = 0.0; g[1, :] = 0.0; return g

    def _hook_bias(grad):
        g = grad.clone(); g[0] = 0.0; g[1] = 0.0; return g

    def _hook_log_std(grad):
        g = grad.clone(); g[0] = 0.0; g[1] = 0.0; return g

    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    model.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layers 2+3: action_net rows 0,1 + log_std[0,1] (gradient hooks)")

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"[Freeze] Trainable: {trainable:,} / {total:,} params (131 policy params)")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d35", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D35 — Seal the Second Pit: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:  d34 (85% random completion, features reduced -13.5%)")
    print(f"        W_pit[128]:     {new_w_tl:+.4f}  (3× d32, overcomes features_noise at step ~1314)")
    print(f"        b_pit:          {new_b_pit:+.4f}  (threshold still at tl≈0.60)")
    print(f"        P(pit|tl=0.53): ≈100%  (2nd pit window — was 95% direct in d34)")
    print(f"        P(pit|tl=0.60): ≈ 50%  (threshold)")
    print(f"        P(pit|tl=0.70): ≈  0%  (don't pit on fresh tyres)")
    print(f"        Net at step ~1314: +2.1 + features(-1.5) = +0.6 → fires deterministically!")
    print(f"        Goal: fixed-start pits=2.0, laps≥13, reward>3000")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    # ── Post-training diagnostics ──────────────────────────────────────────────
    w_tl_after      = model.policy.action_net.weight[2, 128].item()
    b_pit_after     = model.policy.action_net.bias[2].item()
    pit_w_128_after = model.policy.action_net.weight[2, :128].abs().mean().item()
    log_std_after   = model.policy.log_std[2].item()
    thr_w_after     = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after     = model.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_after:.6f}  [SHOULD be {d34_thr_w_mean:.6f}]")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_after:.6f}  [SHOULD be {d34_str_w_mean:.6f}]")
    print(f"       action_net.weight[2, :128] abs_mean = {pit_w_128_after:.6f}  [was {d34_pit_w_mean:.6f}]")
    print(f"       action_net.weight[2,  128]      = {w_tl_after:+.4f}   [init {new_w_tl:+.4f}]")
    print(f"       action_net.bias[2]              = {b_pit_after:+.4f}   [init {new_b_pit:+.4f}]")
    print(f"       log_std[2]                      = {log_std_after:.4f}   [was {d34_log_std:.4f}]")

    d34_model = PPO.load(checkpoint_path, device=device)
    feature_drift = max(
        abs(p.data - q.data).max().item()
        for p, q in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            d34_model.policy.mlp_extractor.policy_net.parameters(),
        )
    )
    print(f"\n[Diag] Features drift vs d34: {feature_drift:.2e}  [should be 0.00]")

    print(f"\n[Diag] Pit signal at key tyre_life values (post-training, direct component):")
    pit_std_after = math.exp(log_std_after)
    threshold_tl  = -b_pit_after / w_tl_after if w_tl_after != 0 else float('nan')
    for tl in [0.30, 0.45, 0.53, 0.60, 0.69, 0.80, 0.90]:
        pit_direct = w_tl_after * tl + b_pit_after
        z = -pit_direct / pit_std_after
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        note = "2nd pit window" if tl == 0.53 else ("pit" if tl < 0.60 else "hold")
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{note}]")
    print(f"\n[Diag] Effective threshold: tl ≈ {threshold_tl:.3f}")

    save_path = str(project_root / "rl" / "ppo_pit_v4_d35.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d35 to {save_path}")
    print(f"        Target: fixed-start pits=2.0, laps≥13, reward>3000.")


if __name__ == "__main__":
    train()
