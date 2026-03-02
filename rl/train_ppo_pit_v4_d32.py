"""
PPO Pit Strategy v4 D32 — Direct Tyre-Life Connection (Week 5).

WHY D32?
=========
D26–D31 all failed because the linear pit row + frozen mlp_extractor features
cannot achieve state-conditional pitting:

  D26–D29: Forced-pit gradient goes wrong direction (bias: +0.006 → -1.217).
  D30:     voluntary_pit_reward correct in principle, but 50% random pit rate
           swamps the +300 bonus with fresh-tyre -200 penalties.
  D31:     BC pre-training of pit row fails — frozen features separate worn/fresh
           states by only 3.25% of required separation. BC loss: 2.55 (target: 0.0).

ROOT CAUSE: The frozen mlp_extractor was trained for throttle/steer, not pit timing.
  Its features do NOT linearly encode tyre_life in a way the pit row can exploit.

THE D32 FIX — DIRECT TYRE_LIFE → PIT_SIGNAL CONNECTION:
  Use PitAwarePolicy (rl/pit_aware_policy.py) which augments the actor's latent
  representation with obs[11] (tyre_life) DIRECTLY:

      latent_pi = [mlp_extractor.policy_net(obs) | obs[11]]  (129-dim)
      pit_signal = latent_pi @ W_pit + b_pit

  The action_net is now Linear(129, 3) instead of Linear(128, 3).
  The 129th weight provides a DIRECT linear path from tyre_life to pit_signal,
  completely bypassing the feature bottleneck.

INITIALIZATION (guarantees state-conditional pitting from episode 1):
  W_pit[128] = -10.0   (higher tyre_life → more negative → don't pit)
  b_pit      = +7.0    (effective pit threshold at tyre_life ≈ 0.69)

  At initialization (ignoring small features noise):
    tl=0.30: pit_signal ≈ +4.0,  P(pit>0) ≈ 97%  ← always pit (critical)
    tl=0.45: pit_signal ≈ +2.5,  P(pit>0) ≈ 88%  ← usually pit (worn)
    tl=0.60: pit_signal ≈ +1.0,  P(pit>0) ≈ 68%  ← often pit (voluntary bonus zone)
    tl=0.69: pit_signal ≈  0.0,  P(pit>0) ≈ 50%  ← coin flip (threshold)
    tl=0.80: pit_signal ≈ -1.0,  P(pit>0) ≈ 32%  ← rarely pit (fresh)
    tl=0.90: pit_signal ≈ -2.0,  P(pit>0) ≈ 17%  ← almost never pit (new tyres)

THREE-LAYER FREEZE (same as d26-d31, extended for 129-dim action_net):
  1. mlp_extractor.policy_net: requires_grad=False (18,176 params)
  2. action_net rows 0,1 (throttle, steer) via gradient hooks — all 129 cols frozen
  3. log_std dims 0,1 via gradient hook

TRAINABLE (131 policy params):
  action_net.weight[2, :] (129 values: 128 feature weights + 1 tyre_life weight)
  action_net.bias[2]      (1 value)
  log_std[2]              (1 value)

PPO ENVIRONMENT: make_env_pit_d30 (voluntary_pit_reward=True, no forced pit)
  +300 bonus when pit_signal > 0 AND tyre_life < 0.60.
  Net voluntary pit cost: -200 + 300 = +100 (immediately profitable when correct).

EXPECTED BEHAVIOUR:
  From episode 1: agent pits ~88% when tyre_life < 0.45, ~17% when tyre_life > 0.90.
  voluntary_pit_reward fires on worn-tyre pits → POSITIVE advantage → PPO reinforces
  high pit_signal at worn states. Fresh-tyre pits (low probability) generate -200 but
  rarely → minimal negative gradient. Net gradient direction: POSITIVE (finally!).

STARTING POINT: ppo_pit_v4.zip (d21, reward=1877, 7 laps, 1 pit, bias=+0.006)
SAVES TO:       rl/ppo_pit_v4_d32.zip
LOGS TO:        runs/ppo_pit_v4_d32/
"""

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
from rl.pit_aware_policy import PitAwarePolicy, copy_d21_weights_into_pit_aware
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21, reward=1877, bias=+0.006)")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit_d30])    # voluntary_pit_reward=True, no forced pit

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    print(f"\n[Train] Loading d21 checkpoint for weight extraction...")
    model_d21 = PPO.load(checkpoint_path, device=device)

    # Pre-load diagnostics from d21
    d21_pit_bias    = model_d21.policy.action_net.bias[2].item()
    d21_pit_w_mean  = model_d21.policy.action_net.weight[2, :].abs().mean().item()
    d21_log_std     = model_d21.policy.log_std[2].item()
    d21_thr_w_mean  = model_d21.policy.action_net.weight[0, :].abs().mean().item()
    d21_str_w_mean  = model_d21.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] D21 reference weights:")
    print(f"       action_net.weight[0,:] abs_mean = {d21_thr_w_mean:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {d21_str_w_mean:.6f}  (steer)")
    print(f"       action_net.weight[2,:] abs_mean = {d21_pit_w_mean:.6f}  (pit)")
    print(f"       action_net.bias[2]     = {d21_pit_bias:+.6f}  (pit bias — will be REPLACED with +7.0)")
    print(f"       log_std[2]             = {d21_log_std:.6f}  (pit std = {torch.exp(torch.tensor(d21_log_std)).item():.3f})")

    # ── Create PitAwarePolicy model ───────────────────────────────────────────
    print(f"\n[Build] Creating PitAwarePolicy model (129-dim action_net input)...")
    model_d32 = PPO(
        policy=PitAwarePolicy,
        env=env,
        learning_rate=cosine_schedule(initial_lr=1e-4, min_lr=1e-6),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
        device=device,
        verbose=1,
    )

    # Verify architecture
    action_net_in  = model_d32.policy.action_net.in_features
    action_net_out = model_d32.policy.action_net.out_features
    print(f"[Build] action_net: Linear({action_net_in}, {action_net_out})  ← should be (129, 3)")
    assert action_net_in == 129, f"Expected 129, got {action_net_in}"
    assert action_net_out == 3,  f"Expected 3, got {action_net_out}"
    print(f"[Build] Architecture verified: action_net is Linear(129, 3) ✓")

    # ── Copy d21 weights into PitAwarePolicy ──────────────────────────────────
    print(f"\n[Init] Copying d21 weights into PitAwarePolicy model...")
    copy_d21_weights_into_pit_aware(model_d21, model_d32)

    # Verify pit signal initialization
    print(f"\n[Init] Post-copy diagnostics:")
    w_tl = model_d32.policy.action_net.weight[2, 128].item()
    b_pit = model_d32.policy.action_net.bias[2].item()
    log_std_pit = model_d32.policy.log_std[2].item()
    w_pit_128 = model_d32.policy.action_net.weight[2, :128].abs().mean().item()
    print(f"       action_net.weight[2, :128] abs_mean = {w_pit_128:.6f}  (copied from d21)")
    print(f"       action_net.weight[2,  128]          = {w_tl:+.4f}   (NEW: tyre_life weight)")
    print(f"       action_net.bias[2]                  = {b_pit:+.4f}   (NEW: recalibrated)")
    print(f"       log_std[2]                          = {log_std_pit:.4f}   (copied from d21)")
    print(f"       action_net.weight[0, 128]           = {model_d32.policy.action_net.weight[0, 128].item():+.4f}   (throttle tl: should be 0)")
    print(f"       action_net.weight[1, 128]           = {model_d32.policy.action_net.weight[1, 128].item():+.4f}   (steer tl: should be 0)")

    # ── LAYER 1: Freeze mlp_extractor.policy_net ──────────────────────────────
    frozen_feat = 0
    for param in model_d32.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Freeze throttle/steer rows via gradient hooks ────────────────
    # action_net.weight is now (3, 129): freeze rows 0 and 1 (all 129 cols)

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0, :] = 0.0    # freeze throttle row (all 129 cols including tyre_life)
        g[1, :] = 0.0    # freeze steer row    (all 129 cols including tyre_life)
        return g

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model_d32.policy.action_net.weight.register_hook(_hook_weight)
    model_d32.policy.action_net.bias.register_hook(_hook_bias)
    print(f"[Freeze] Layer 2: action_net.weight rows [0,1] + bias[0,1] (gradient hooks, 129-dim rows)")

    # ── LAYER 3: Freeze log_std throttle/steer dims ──────────────────────────

    def _hook_log_std(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model_d32.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layer 3: log_std[0,1] (gradient hook)")

    # Print summary
    trainable = sum(p.numel() for p in model_d32.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model_d32.policy.parameters())
    print(f"\n[Freeze] Summary:")
    print(f"         Frozen (features):    {frozen_feat:,} params")
    print(f"         Frozen (hook):        action_net rows 0,1 + log_std[0,1]")
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 131 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── Set lr_schedule (both attributes, d29 bug fix) ────────────────────────
    # Already set at PPO construction via learning_rate=cosine_schedule(...)
    # But model.lr_schedule is also set by SB3 from the learning_rate argument.
    # Verify both are set correctly:
    lr_at_start = model_d32.lr_schedule(1.0)
    print(f"\n[LR] cosine(1e-4 → 1e-6), lr_schedule(1.0) = {lr_at_start:.2e}  (should be 1e-4)")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model_d32.set_logger(configure("runs/ppo_pit_v4_d32", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D32: {TOTAL_STEPS:,} steps")
    print(f"        Policy:              PitAwarePolicy (129-dim action_net)")
    print(f"        Action net:          Linear(129, 3) — 128 features + 1 tyre_life")
    print(f"        Starting from:       d21 weights + direct tyre_life connection")
    print(f"        W_pit[128]:          -10.0  (direct tyre_life → pit)")
    print(f"        b_pit:               +7.0   (threshold at tl ≈ 0.69)")
    print(f"        P(pit|tl=0.45):      ≈88%  (worn → pits often)")
    print(f"        P(pit|tl=0.90):      ≈17%  (fresh → rarely pits)")
    print(f"        Environment:         make_env_pit_d30 (voluntary_pit_reward=True)")
    print(f"        voluntary_pit_bonus: +300 when pit_signal>0 AND tyre_life<0.60")
    print(f"        Frozen:              mlp_extractor.policy_net + throttle/steer rows")
    print(f"        Trainable policy:    131 params (pit row only, now 129-dim)")
    print(f"        Goal: pit_signal state-conditional from episode 1, reward > 2000")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model_d32.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    w_tl_after    = model_d32.policy.action_net.weight[2, 128].item()
    b_pit_after   = model_d32.policy.action_net.bias[2].item()
    pit_w_128_after = model_d32.policy.action_net.weight[2, :128].abs().mean().item()
    log_std_after = model_d32.policy.log_std[2].item()
    thr_w_after   = model_d32.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after   = model_d32.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_after:.6f}  [SHOULD be {d21_thr_w_mean:.6f}]")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_after:.6f}  [SHOULD be {d21_str_w_mean:.6f}]")
    print(f"       action_net.weight[2, :128] abs_mean = {pit_w_128_after:.6f}  [was {d21_pit_w_mean:.6f}]")
    print(f"       action_net.weight[2,  128]      = {w_tl_after:+.4f}   [was -10.0000 — did it move?]")
    print(f"       action_net.bias[2]              = {b_pit_after:+.4f}   [was +7.0000 — did it move?]")
    print(f"       log_std[2]                      = {log_std_after:.4f}   [was {d21_log_std:.4f}]")

    # Verify freeze quality vs d21
    feature_drift = max(
        (p_new - p_old).abs().max().item()
        for p_new, p_old in zip(
            model_d32.policy.mlp_extractor.policy_net.parameters(),
            model_d21.policy.mlp_extractor.policy_net.parameters(),
        )
    )
    thr_drift = model_d32.policy.action_net.weight[0, :128].sub(
        model_d21.policy.action_net.weight[0, :]
    ).abs().max().item()
    str_drift = model_d32.policy.action_net.weight[1, :128].sub(
        model_d21.policy.action_net.weight[1, :]
    ).abs().max().item()

    print(f"\n[Diag] Freeze verification vs d21 (all should be ≈ 0.0):")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features]")
    print(f"       action_net.weight[0,:128] drift = {thr_drift:.2e}  [throttle row]")
    print(f"       action_net.weight[1,:128] drift = {str_drift:.2e}  [steer row]")

    # Key diagnosis: did the tyre_life weight stay effective?
    print(f"\n[Diag] Pit signal at key tyre_life values (post-training):")
    for tl in [0.30, 0.45, 0.60, 0.69, 0.80, 0.90]:
        # Direct component only (features noise excluded)
        pit_direct = w_tl_after * tl + b_pit_after
        import math
        pit_std = math.exp(log_std_after)
        z = -pit_direct / pit_std
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        target = "pit" if tl < 0.60 else "hold"
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{target}]")

    if w_tl_after < -5.0:
        print(f"\n[Diag] → Tyre_life weight remains strongly negative ({w_tl_after:.2f}) ✓")
        print(f"         State-conditional pitting PRESERVED by direct connection.")
    else:
        print(f"\n[Diag] → Tyre_life weight weakened ({w_tl_after:.2f}) — PPO adjusted the threshold.")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d32.zip")
    model_d32.save(save_path)
    print(f"\n[Train] Saved d32 model to {save_path}")
    print(f"        Policy class: PitAwarePolicy (auto-reconstructed on PPO.load)")
    print(f"        Run evaluate.py to compare against d21 (1877) and d31 (1879).")
    print(f"        Target: reward > 2000 (voluntary pit at tyre_life < 0.60).")


if __name__ == "__main__":
    train()
