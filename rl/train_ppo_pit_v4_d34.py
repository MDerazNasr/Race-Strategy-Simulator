"""
PPO Pit Strategy v4 D34 — Two-Pit Strategy Take 2 (Week 5).

WHY D33 FAILED:
================
D33 continued training from d32 with LR cosine 1e-5→1e-7 (10× lower than d32).
The hypothesis was that d32's long episodes (step 2000) would expose the
second-pit window (step ~1314, tl≈0.53) and PPO would reinforce it.

Result: IDENTICAL to d32 (reward=2683.83, 11 laps, 1 pit). Weight barely moved:
  W_pit[128]: -10.4882 → -10.4927  (<0.05% change)
  approx_kl collapsed to 1e-9 — essentially frozen.

ROOT CAUSE — SECOND FEATURE BOTTLENECK:
  At the second-pit window (step ~1314, tl≈0.53):
    pit_signal = features_contribution + W_pit[128] × tl + b_pit
               = features_contribution + (-10.49 × 0.53) + 6.50
               = features_contribution + 0.94
  The frozen features at that late-episode state give features_contribution ≈ -1.5.
  Net deterministic pit_signal ≈ -0.56 → NEGATIVE → no pit fires.

  W_pit[128] needs to grow from -10.49 to approximately -20 so that:
    -20 × 0.53 + b = -10.6 + b → needs b ≈ 12 to give +1.40 (overcomes -1.5)
  LR of 1e-5 caused <0.05% change — completely insufficient.

THE D34 FIX:
  1. Load d32 weights (driving + first-pit calibration)
  2. RE-INITIALIZE pit row: W_pit[128] = -20.0, b_pit = +12.0
       Threshold preserved: -20 × 0.60 + 12 = 0 → pit threshold still at tl=0.60
       Second-pit window:   -20 × 0.53 + 12 = +1.40 → overcomes -1.5 features ✓
  3. Train with LR cosine 1e-4 → 1e-6 (same as d32, NOT the too-low 1e-5 of d33)

Pit probabilities after re-initialization (direct component + std=0.864):
  tl=0.30: pit_direct = -20×0.30+12 = +6.0   P(pit>0) ≈ 100%  [always pit]
  tl=0.45: pit_direct = -20×0.45+12 = +3.0   P(pit>0) ≈ 100%  [always pit]
  tl=0.53: pit_direct = -20×0.53+12 = +1.40  P(pit>0) ≈  95%  [2nd pit window ✓]
  tl=0.60: pit_direct = -20×0.60+12 =  0.0   P(pit>0) ≈  50%  [threshold]
  tl=0.70: pit_direct = -20×0.70+12 = -2.0   P(pit>0) ≈   1%  [hold]
  tl=0.90: pit_direct = -20×0.90+12 = -6.0   P(pit>0) ≈   0%  [never pit]

STARTING POINT: ppo_pit_v4_d32.zip (d32, reward=2683, 11 laps, 1 pit)
  - Load d32 (keeps driving + first-pit features)
  - Override W_pit[128] and b_pit with new larger values
  - Keep log_std[2] = -0.1462 (std=0.864 from d32)
  - Keep W_pit[:128] from d32 (existing feature weights)

SAVES TO: rl/ppo_pit_v4_d34.zip
LOGS TO:  runs/ppo_pit_v4_d34/
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

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d32.zip")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4_d32.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4_d32.py (d32) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d32, reward=2683, 11 laps, 1 pit)")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit_d30])    # voluntary_pit_reward=True, no forced pit

    # ── Load d32 checkpoint ───────────────────────────────────────────────────
    print(f"\n[Train] Loading d32 checkpoint (PitAwarePolicy, 129-dim action_net)...")
    model = PPO.load(checkpoint_path, env=env, device=device)

    # Pre-load diagnostics from d32
    d32_w_tl       = model.policy.action_net.weight[2, 128].item()
    d32_b_pit      = model.policy.action_net.bias[2].item()
    d32_log_std    = model.policy.log_std[2].item()
    d32_pit_w_mean = model.policy.action_net.weight[2, :128].abs().mean().item()
    d32_thr_w_mean = model.policy.action_net.weight[0, :].abs().mean().item()
    d32_str_w_mean = model.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] D32 loaded weights (before re-initialization):")
    print(f"       action_net.weight[0,:] abs_mean = {d32_thr_w_mean:.6f}  (throttle — frozen)")
    print(f"       action_net.weight[1,:] abs_mean = {d32_str_w_mean:.6f}  (steer — frozen)")
    print(f"       action_net.weight[2,:128] abs_mean = {d32_pit_w_mean:.6f}  (pit features — kept)")
    print(f"       action_net.weight[2, 128]        = {d32_w_tl:+.4f}   (tyre_life direct: will be overridden)")
    print(f"       action_net.bias[2]               = {d32_b_pit:+.4f}   (pit bias: will be overridden)")
    print(f"       log_std[2]                       = {d32_log_std:.4f}   (pit std = {torch.exp(torch.tensor(d32_log_std)).item():.3f}, kept)")

    # Verify architecture
    action_net_in  = model.policy.action_net.in_features
    action_net_out = model.policy.action_net.out_features
    print(f"\n[Build] action_net: Linear({action_net_in}, {action_net_out})  ← should be (129, 3)")
    assert action_net_in == 129, f"Expected 129, got {action_net_in}"
    assert action_net_out == 3,  f"Expected 3, got {action_net_out}"
    print(f"[Build] Architecture verified: action_net is Linear(129, 3) ✓")

    # ── RE-INITIALIZE pit row with larger direct weight ────────────────────────
    # W_pit[128] = -20.0  (was -10.49 in d32)
    # b_pit      = +12.0  (was  +6.50 in d32)
    #
    # Threshold preservation: -20 × 0.60 + 12 = 0  → pit threshold still tl=0.60 ✓
    # Second-pit window:      -20 × 0.53 + 12 = +1.40 → overcomes features_noise ≈ -1.5 ✓
    #
    # Keep W_pit[:128] from d32 (feature weights from d32 training are kept).
    with torch.no_grad():
        model.policy.action_net.weight[2, 128] = -20.0
        model.policy.action_net.bias[2]        = +12.0

    new_w_tl  = model.policy.action_net.weight[2, 128].item()
    new_b_pit = model.policy.action_net.bias[2].item()
    print(f"\n[Init] Re-initialized pit direct connection:")
    print(f"       W_pit[128]: {d32_w_tl:+.4f} → {new_w_tl:+.4f}  (2× larger magnitude)")
    print(f"       b_pit:      {d32_b_pit:+.4f} → {new_b_pit:+.4f}  (threshold preserved at tl≈0.60)")

    # Sanity check pit signals after re-init
    pit_std = math.exp(d32_log_std)
    print(f"\n[Init] Post-init pit probabilities (direct component, std={pit_std:.3f}):")
    for tl in [0.30, 0.45, 0.53, 0.60, 0.70, 0.80, 0.90]:
        pit_direct = new_w_tl * tl + new_b_pit
        z = -pit_direct / pit_std
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        note = "2nd pit window" if tl == 0.53 else ("pit" if tl < 0.60 else "hold")
        check = "✓" if (tl < 0.60 and p_pit > 0.85) or (tl >= 0.60 and p_pit < 0.15) else "!"
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{note}] {check}")

    # ── Set LR schedule (same as d32: 1e-4 → 1e-6) ───────────────────────────
    new_lr_schedule = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_lr_schedule
    model.lr_schedule   = new_lr_schedule
    lr_at_start = model.lr_schedule(1.0)
    print(f"\n[LR] cosine(1e-4 → 1e-6), lr_schedule(1.0) = {lr_at_start:.2e}  (same as d32)")

    # ── LAYER 1: Re-freeze mlp_extractor.policy_net ───────────────────────────
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Re-freeze throttle/steer rows via gradient hooks ─────────────

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0, :] = 0.0
        g[1, :] = 0.0
        return g

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    print(f"[Freeze] Layer 2: action_net.weight rows [0,1] + bias[0,1] (gradient hooks)")

    # ── LAYER 3: Re-freeze log_std throttle/steer dims ────────────────────────

    def _hook_log_std(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layer 3: log_std[0,1] (gradient hook)")

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Freeze] Trainable (policy): 131 params (pit row + bias + log_std[2])")
    print(f"         Trainable (total):  {trainable:,} / {total:,} params")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d34", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D34 — Two-Pit Take 2: {TOTAL_STEPS:,} steps")
    print(f"        Policy:              PitAwarePolicy (d32 weights + re-init pit row)")
    print(f"        W_pit[128]:          {new_w_tl:+.4f}  (was {d32_w_tl:+.4f} — 2× larger)")
    print(f"        b_pit:               {new_b_pit:+.4f}  (threshold still at tl≈0.60)")
    print(f"        P(pit|tl=0.53):      ≈95%  (2nd pit window — was 26% effective)")
    print(f"        P(pit|tl=0.60):      ≈50%  (threshold)")
    print(f"        P(pit|tl=0.70):      ≈ 1%  (don't pit on fresh tyres)")
    print(f"        LR:                  cosine(1e-4 → 1e-6) — same as d32")
    print(f"        Environment:         make_env_pit_d30 (voluntary_pit_reward=True)")
    print(f"        Goal: 2 pits, 13-14 laps, reward > 3000")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    w_tl_after      = model.policy.action_net.weight[2, 128].item()
    b_pit_after     = model.policy.action_net.bias[2].item()
    pit_w_128_after = model.policy.action_net.weight[2, :128].abs().mean().item()
    log_std_after   = model.policy.log_std[2].item()
    thr_w_after     = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after     = model.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_after:.6f}  [SHOULD be {d32_thr_w_mean:.6f}]")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_after:.6f}  [SHOULD be {d32_str_w_mean:.6f}]")
    print(f"       action_net.weight[2, :128] abs_mean = {pit_w_128_after:.6f}  [was {d32_pit_w_mean:.6f}]")
    print(f"       action_net.weight[2,  128]      = {w_tl_after:+.4f}   [was {new_w_tl:+.4f} at init]")
    print(f"       action_net.bias[2]              = {b_pit_after:+.4f}   [was {new_b_pit:+.4f} at init]")
    print(f"       log_std[2]                      = {log_std_after:.4f}   [was {d32_log_std:.4f}]")

    # Verify freeze quality
    d32_model = PPO.load(checkpoint_path, device=device)
    feature_drift = max(
        abs(p.data - q.data).max().item()
        for p, q in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            d32_model.policy.mlp_extractor.policy_net.parameters(),
        )
    )
    print(f"\n[Diag] Freeze verification vs d32:")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [should be 0.00]")

    print(f"\n[Diag] Pit signal at key tyre_life values (post-training, direct component):")
    pit_std_after = math.exp(log_std_after)
    for tl in [0.30, 0.45, 0.53, 0.60, 0.69, 0.80, 0.90]:
        pit_direct = w_tl_after * tl + b_pit_after
        z = -pit_direct / pit_std_after
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        note = "2nd pit window" if tl == 0.53 else ("pit" if tl < 0.60 else "hold")
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{note}]")

    threshold_tl = -b_pit_after / w_tl_after if w_tl_after != 0 else float('nan')
    print(f"\n[Diag] Effective pit threshold: tl ≈ {threshold_tl:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d34.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d34 model to {save_path}")
    print(f"        Run evaluate.py to compare against d32 (2683, 11 laps, 1 pit).")
    print(f"        Target: 2 pits, 13-14 laps, reward > 3000.")


if __name__ == "__main__":
    train()
