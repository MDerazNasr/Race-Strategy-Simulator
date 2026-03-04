"""
PPO Pit Strategy v4 D33 — Two-Pit Strategy (Week 5).

WHY D33?
=========
D32 (PitAwarePolicy) broke the 1883 fixed-start ceiling with ONE voluntary pit
at tyre_life ≈ 0.55–0.60 (reward=2683, 11 laps, 100% completion, step 2000).

The direct tyre_life → pit_signal connection means the same weights SHOULD
re-trigger whenever tyre_life drops below 0.60 a second time — but d32
evaluation shows pits=1.0 (exactly 1 pit, fixed start). The second pit does
not fire.

ROOT CAUSE — D32's TRAINING DISTRIBUTION:
  During d32 training, episodes were short (ep_len ≈ 800 steps average).
  The agent almost never reached the second-pit window (~step 1314 in a
  fixed-start episode). As a result:
    - The features[:128] @ W_pit[:128] component was shaped by mid-episode
      worn-tyre states (step ~300–650), and likely encodes context different
      from late-episode worn-tyre states (step ~1100–1400).
    - At the second-pit window (tl≈0.53, step ~1314): direct component =
      −10.49×0.53+6.50 = +0.94, but features noise ≈ −1.5 → net ≈ −0.56.
      P(pit>0) ≈ 26% — not enough to trigger reliably.
    - No gradient ever reinforced the second pit because it was rarely seen.

THE D33 FIX — MORE TRAINING FROM D32:
  D32 already survives to step 2000 (fixed-start). During d33 training,
  episodes will regularly be 1500–2000 steps. The agent will frequently
  encounter the second-pit window (tl<0.60 at step ~1314). PPO will:
    1. See: second pit fires occasionally (P≈26%) → +100 net reward
    2. Positive advantage → W_pit[128] increases in magnitude (more negative)
       to push pit_signal more positive at worn states
    3. After several updates, second pit fires reliably → 13–14 laps

SECOND-PIT TIMELINE (tyre mechanics):
  Step 0:   tl=1.0 (fresh tyres), decay rate ≈ 0.0007/step
  Step ~643: tl≈0.55 → voluntary pit fires (first pit), tl resets to 1.0
  Step ~743: pit cooldown expires (100 steps)
  Step ~1314: tl≈0.53 → second pit opportunity (should fire with d33 training)
  Step ~1414: pit cooldown expires again
  Step 2000: episode end — with fresh tyres from step 1314, 2–3 more laps

ARCHITECTURE: PitAwarePolicy (unchanged from d32)
  - TyrLifeAugmentedExtractor: latent_dim_pi=129 (128 features + tyre_life)
  - action_net = Linear(129, 3) — W_pit[128] = direct tyre_life connection
  - Same three-layer freeze: features frozen, throttle/steer rows frozen
  - 131 trainable params (pit row 129 + bias + log_std[2])

STARTING POINT: ppo_pit_v4_d32.zip (d32, reward=2683, 11 laps, 1 pit)
  - Load directly (already a PitAwarePolicy — no weight copy needed)
  - Re-apply three-layer freeze (gradient hooks not saved by SB3)
  - Set new LR schedule: cosine 1e-5 → 1e-7 (10× lower than d32's 1e-4→1e-6)
    Lower LR preserves first-pit calibration while PPO discovers second pit.

SAVES TO:       rl/ppo_pit_v4_d33.zip
LOGS TO:        runs/ppo_pit_v4_d33/
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
    d32_w_tl        = model.policy.action_net.weight[2, 128].item()
    d32_b_pit       = model.policy.action_net.bias[2].item()
    d32_log_std     = model.policy.log_std[2].item()
    d32_pit_w_mean  = model.policy.action_net.weight[2, :128].abs().mean().item()
    d32_thr_w_mean  = model.policy.action_net.weight[0, :].abs().mean().item()
    d32_str_w_mean  = model.policy.action_net.weight[1, :].abs().mean().item()

    print(f"\n[Diag] D32 starting weights (will be preserved / fine-tuned):")
    print(f"       action_net.weight[0,:] abs_mean = {d32_thr_w_mean:.6f}  (throttle — frozen)")
    print(f"       action_net.weight[1,:] abs_mean = {d32_str_w_mean:.6f}  (steer — frozen)")
    print(f"       action_net.weight[2,:128] abs_mean = {d32_pit_w_mean:.6f}  (pit features)")
    print(f"       action_net.weight[2, 128]        = {d32_w_tl:+.4f}   (tyre_life direct: should ≈ -10.49)")
    print(f"       action_net.bias[2]               = {d32_b_pit:+.4f}   (pit bias: should ≈ +6.50)")
    print(f"       log_std[2]                       = {d32_log_std:.4f}   (pit std = {torch.exp(torch.tensor(d32_log_std)).item():.3f})")

    # Verify architecture
    action_net_in  = model.policy.action_net.in_features
    action_net_out = model.policy.action_net.out_features
    print(f"\n[Build] action_net: Linear({action_net_in}, {action_net_out})  ← should be (129, 3)")
    assert action_net_in == 129, f"Expected 129, got {action_net_in}"
    assert action_net_out == 3,  f"Expected 3, got {action_net_out}"
    print(f"[Build] Architecture verified: action_net is Linear(129, 3) ✓")

    # ── Set new LR schedule (lower than d32: 1e-5 → 1e-7) ────────────────────
    # SB3 PPO.load() restores the old lr_schedule; we override for d33.
    # 10× lower than d32 (1e-4→1e-6) to preserve first-pit calibration.
    new_lr_schedule = cosine_schedule(initial_lr=1e-5, min_lr=1e-7)
    model.learning_rate = new_lr_schedule
    model.lr_schedule   = new_lr_schedule
    lr_at_start = model.lr_schedule(1.0)
    print(f"\n[LR] cosine(1e-5 → 1e-7), lr_schedule(1.0) = {lr_at_start:.2e}  (should be 1e-5)")

    # ── LAYER 1: Re-freeze mlp_extractor.policy_net ───────────────────────────
    # Gradient hooks are NOT preserved by SB3 save/load — must re-apply.
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Re-freeze throttle/steer rows via gradient hooks ─────────────
    # action_net.weight is (3, 129): freeze rows 0 and 1 (all 129 cols)

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0, :] = 0.0    # freeze throttle row (all 129 cols)
        g[1, :] = 0.0    # freeze steer row    (all 129 cols)
        return g

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    print(f"[Freeze] Layer 2: action_net.weight rows [0,1] + bias[0,1] (gradient hooks, 129-dim rows)")

    # ── LAYER 3: Re-freeze log_std throttle/steer dims ────────────────────────

    def _hook_log_std(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0
        g[1] = 0.0
        return g

    model.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layer 3: log_std[0,1] (gradient hook)")

    # Print summary
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Freeze] Summary:")
    print(f"         Frozen (features):    {frozen_feat:,} params")
    print(f"         Frozen (hook):        action_net rows 0,1 + log_std[0,1]")
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 131 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d33", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D33 — Two-Pit Strategy: {TOTAL_STEPS:,} steps")
    print(f"        Policy:              PitAwarePolicy (129-dim action_net, loaded from d32)")
    print(f"        Starting from:       d32 weights (reward=2683, 11 laps, 1 pit)")
    print(f"        W_pit[128]:          {d32_w_tl:+.4f}  (direct tyre_life → pit)")
    print(f"        b_pit:               {d32_b_pit:+.4f}  (threshold at tl ≈ 0.62)")
    print(f"        LR:                  cosine(1e-5 → 1e-7) — 10× lower than d32")
    print(f"        Goal:                second pit at tl<0.60, step ~1314 → 13-14 laps")
    print(f"        Environment:         make_env_pit_d30 (voluntary_pit_reward=True)")
    print(f"        voluntary_pit_bonus: +300 when pit_signal>0 AND tyre_life<0.60")
    print(f"        Why second pit will emerge: d32 episodes reach 2000 steps → agent")
    print(f"          now regularly sees the second-pit window (step ~1314, tl≈0.53).")
    print(f"          PPO receives +100 net reward when second pit fires → reinforces.")
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
    print(f"       action_net.weight[2,  128]      = {w_tl_after:+.4f}   [was {d32_w_tl:+.4f} — did it grow?]")
    print(f"       action_net.bias[2]              = {b_pit_after:+.4f}   [was {d32_b_pit:+.4f}]")
    print(f"       log_std[2]                      = {log_std_after:.4f}   [was {d32_log_std:.4f}]")

    # Verify freeze quality vs d32
    feature_drift = max(
        abs(p.data - q.data).max().item()
        for p, q in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            PPO.load(checkpoint_path, device=device).policy.mlp_extractor.policy_net.parameters(),
        )
    )
    print(f"\n[Diag] Freeze verification vs d32:")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features — should be 0.00]")

    # Pit signal at key tyre_life values
    print(f"\n[Diag] Pit signal at key tyre_life values (post-training, direct component):")
    for tl in [0.30, 0.45, 0.53, 0.60, 0.69, 0.80, 0.90]:
        import math
        pit_direct = w_tl_after * tl + b_pit_after
        pit_std = math.exp(log_std_after)
        z = -pit_direct / pit_std
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        note = "2nd pit window" if tl == 0.53 else ("pit" if tl < 0.60 else "hold")
        print(f"       tl={tl:.2f}: pit_direct≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{note}]")

    if w_tl_after < -10.0:
        print(f"\n[Diag] → Tyre_life weight grew more negative ({w_tl_after:.2f}) ✓")
        print(f"         Second-pit signal strengthened — good for two-pit strategy!")
    elif w_tl_after < -5.0:
        print(f"\n[Diag] → Tyre_life weight unchanged ({w_tl_after:.2f}) — check if second pit fires in eval.")
    else:
        print(f"\n[Diag] → Tyre_life weight weakened ({w_tl_after:.2f}) — may need investigation.")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d33.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d33 model to {save_path}")
    print(f"        Policy class: PitAwarePolicy (auto-reconstructed on PPO.load)")
    print(f"        Run evaluate.py to compare against d32 (2683, 11 laps, 1 pit).")
    print(f"        Target: 2 pits, 13-14 laps, reward > 3000.")


if __name__ == "__main__":
    train()
