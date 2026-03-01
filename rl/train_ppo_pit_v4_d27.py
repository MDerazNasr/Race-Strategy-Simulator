"""
PPO Pit Strategy v4 D27 — Three-Layer Freeze + Pit Timing Reward (Week 5).

WHY D27?
=========
D26 proved that three-layer freeze works:
  - Driving: IDENTICAL to d21 (all driving params frozen, verified by 0.000 drift)
  - Pitting: MAINTAINED (0.70 avg pits random start, 1883 reward fixed start)
  - Freeze confirmed: features drift=0.000, throttle row drift_max=6.1e-4, steer=7.4e-4

D26 used standard env (no pit_timing_reward) because in d24, loading d21 into
pit_timing_reward env caused a value recalibration crisis that destroyed driving:
  ep_rew_mean → -1470, steer collapsed to -0.74 hard-left, 23-step crash.

BUT D26 changed the picture entirely:
  - All driving params are NOW FROZEN. The three-layer freeze applies at load time.
  - Even if the value function recalibrates (tolerable), the POLICY CANNOT change.
  - The steer=hard-left collapse from d24 CANNOT happen: steer row is hook-frozen.
  - Only the pit row (130 params) and value function can adapt.

THE D27 FIX — ADD PIT TIMING REWARD:
======================================

STARTING POINT: ppo_pit_v4_d26.zip (d26, reward=1883, driving=d21, pit row adapted)
  - d26's pit row is already adapted (weight abs_mean 0.123→0.301, bias 0.006→-0.817)
  - Loading d26 is better than d21: the pit row has been shaped by 2M steps already

ENVIRONMENT: make_env_pit_d23 (pit_timing_reward=True)
  - Pit timing reward: tyre_life < 0.30 when pitting → +100 bonus (net cost = -100)
  - Pit too early (tyre_life > 0.50): -100 extra penalty (net cost = -300)
  - Neutral zone (0.30–0.50): no change (net cost = -200)

  Why this is now safe:
    1. Three-layer freeze applied immediately at load time (before any gradient flows)
    2. The value function WILL recalibrate (it can now "see" the +100 bonus)
    3. But driving CANNOT regress — steer, throttle, features are all frozen
    4. Only pit_row (130 params) can respond to the timing gradient

THREE-LAYER FREEZE — IDENTICAL TO D26:
  Layer 1 — mlp_extractor.policy_net (features): requires_grad=False
  Layer 2 — action_net.weight rows 0,1 + bias[0,1]: gradient hooks
  Layer 3 — log_std[0,1]: gradient hook
  Only 130 policy params trainable: action_net.weight[2,:] + bias[2] + log_std[2]

EXPECTED PIT TIMING SHIFT:
  D21/D26: pit at tyre_life ≈ 0.35 (neutral zone, net cost = -200)
  D27 target: pit at tyre_life ≈ 0.28-0.30 (bonus zone, net cost = -100)
    → saves 100 reward per pit (extra lap worth of reward)
    → with 1 pit per episode: d27 reward ≈ 1883 + 100 = ~1983

  How: pit_timing_reward provides a positive gradient when tyre_life < 0.30.
  The pit row weight/bias must shift the decision boundary from ~0.35 → ~0.29.
  With 130 trainable params and 2M steps, this shift should be achievable.

CURRICULUM: STAGES_PIT_V5 (same as d26)
  Stage 0: threshold=0.25 (~500k steps) — safety net backup
  Stage 1: threshold=0.15 (~1M steps)   — deep backup
  Stage 2: threshold=0.08 (~500k steps) — theoretical safety net only

  D26's natural voluntary pit: tyre_life ≈ 0.35 (before any threshold fires).
  Safety net at 0.25 = backup only. Agent must learn to push timing to 0.29.

LR SCHEDULE: 1e-4 → 1e-6 cosine (same as d26)
  With pit_timing_reward, the gradient signal to the pit row is richer.
  But 130 params is still small — 1e-4 is appropriate.
  Value function can adapt freely (all value layers trainable).

EXPECTED OUTCOME:
  - Driving: IDENTICAL to d21/d26 (all driving params frozen at d26's values)
  - Pitting: MAINTAINED and potentially IMPROVED timing
  - Timing: shift toward tyre_life ≈ 0.28-0.30 for the +100 bonus
  - Reward: ≥ 1883 (d26) ideally ≈ 1983 (+100 timing bonus)
  - Key success metric: reward > 1883 AND pit_count ≥ 1

COMPARISON TO D26:
  d26: standard env + safety net → preserved d21 (1883 reward, 1 pit)
  d27: pit_timing_reward env + safety net → should improve pit timing

STARTING POINT: ppo_pit_v4_d26.zip (d26)
SAVES TO:       rl/ppo_pit_v4_d27.zip
LOGS TO:        runs/ppo_pit_v4_d27/
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

from rl.make_env import make_env_pit_d23      # pit_timing_reward=True
from rl.curriculum import CurriculumCallback, STAGES_PIT_V5
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4_d26.zip")   # d26 — three-layer freeze already shaped

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4_d26.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4_d26.py (d26) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d26, reward=1883, driving=d21)")

    # ── Build environment ──────────────────────────────────────────────────────
    # make_env_pit_d23(): F1Env(tyre_degradation=True, pit_stops=True, pit_timing_reward=True)
    # pit_timing_reward=True is NOW SAFE: driving is three-layer frozen.
    # Value function will recalibrate (tolerable). Policy CANNOT regress.
    env = DummyVecEnv([make_env_pit_d23])

    # ── Load d26 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Pre-training diagnostics
    pit_w_before = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_before = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_before = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_std_before = model.policy.log_std[2].item()
    thr_std_before = model.policy.log_std[0].item()
    str_std_before = model.policy.log_std[1].item()
    pit_bias_before = model.policy.action_net.bias[2].item()
    print(f"\n[Diag] Pre-training weights (d26's adapted pit row):")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_before:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_before:.6f}  (steer)")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_before:.6f}  (pit)")
    print(f"       action_net.bias[2]     = {pit_bias_before:.6f}  (pit bias)")
    print(f"       log_std[0] = {thr_std_before:.6f} (throttle std)")
    print(f"       log_std[1] = {str_std_before:.6f} (steer std)")
    print(f"       log_std[2] = {pit_std_before:.6f} (pit std)")

    # ── LAYER 1: Freeze mlp_extractor.policy_net (driving features) ────────────
    # Identical to d26: 18,176 params, requires_grad=False.
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Freeze action_net throttle/steer rows via gradient hooks ──────
    # Identical to d26: zero gradient for rows 0,1 of weight and elements 0,1 of bias.

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0, :] = 0.0   # throttle row: no update
        g[1, :] = 0.0   # steer row:    no update
        return g

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0   # throttle bias: no update
        g[1] = 0.0   # steer bias:    no update
        return g

    model.policy.action_net.weight.register_hook(_hook_weight)
    model.policy.action_net.bias.register_hook(_hook_bias)
    print(f"[Freeze] Layer 2: action_net.weight rows [0,1] + bias[0,1] (gradient hooks)")

    # ── LAYER 3: Freeze log_std throttle/steer dims via gradient hook ──────────
    # Identical to d26: zero gradient for dims 0,1 of log_std.

    def _hook_log_std(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[0] = 0.0   # throttle std: frozen
        g[1] = 0.0   # steer std:    frozen
        return g

    model.policy.log_std.register_hook(_hook_log_std)
    print(f"[Freeze] Layer 3: log_std[0,1] (gradient hook)")

    # Print summary
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Freeze] Summary:")
    print(f"         Frozen (features):    {frozen_feat:,} params (requires_grad=False)")
    print(f"         Frozen (hook):        rows 0,1 of action_net + log_std[0,1]")
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 130 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── LR schedule ───────────────────────────────────────────────────────────
    # Same as d26: 1e-4 → 1e-6 cosine. pit_timing_reward provides richer pit
    # gradient, but 130 policy params is still small. 1e-4 is appropriate.
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure curriculum callback ─────────────────────────────────────────
    # STAGES_PIT_V5: safety net at 0.25/0.15/0.08.
    # D26 pits voluntarily at tyre_life≈0.35. With timing reward, target is ~0.29.
    # Safety net provides backup if timing shift causes regression.
    callback = CurriculumCallback(stages=STAGES_PIT_V5, verbose=1)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d27", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D27: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:       ppo_pit_v4_d26.zip (d26, reward=1883)")
    print(f"        Starting LR:         1e-4 (cosine decay to 1e-6 over 2M steps)")
    print(f"        Environment:         make_env_pit_d23 (pit_timing_reward=True!)")
    print(f"        Frozen (Layer 1):    mlp_extractor.policy_net ({frozen_feat:,} params)")
    print(f"        Frozen (Layer 2):    action_net rows 0,1 (throttle/steer) via hooks")
    print(f"        Frozen (Layer 3):    log_std[0,1] (throttle/steer stds) via hooks")
    print(f"        Trainable policy:    action_net.weight[2,:] + bias[2] + log_std[2] (130 params)")
    print(f"        Curriculum:          STAGES_PIT_V5 (safety net: 0.25→0.15→0.08)")
    print(f"        Goal:                reward > 1883 AND pit_count ≥ 1")
    print(f"        Timing target:       pit at tyre_life ≈ 0.29 (+100 bonus vs d26's 0.35)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=False,   # step counter continues from d26's ~3M
        callback=callback,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pit_w_after = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_after = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_after = model.policy.action_net.bias[2].item()
    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_after:.6f}  (was {thr_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_after:.6f}  (was {str_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_after:.6f}  (was {pit_w_before:.6f}) [PIT: can change]")
    print(f"       action_net.bias[2]     = {pit_bias_after:.6f}  (was {pit_bias_before:.6f}) [PIT: can change]")

    # Freeze verification vs d26
    state_d26 = PPO.load(checkpoint_path, device=device)

    feature_drift = max(
        (p_new - p_old).abs().max().item()
        for p_new, p_old in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            state_d26.policy.mlp_extractor.policy_net.parameters()
        )
    )
    thr_drift = (
        model.policy.action_net.weight[0, :] - state_d26.policy.action_net.weight[0, :]
    ).abs().max().item()
    str_drift = (
        model.policy.action_net.weight[1, :] - state_d26.policy.action_net.weight[1, :]
    ).abs().max().item()

    print(f"\n[Diag] Freeze verification vs d26 (all should be ≈ 0.0):")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features]")
    print(f"       action_net.weight[0,:] drift   = {thr_drift:.2e}  [throttle row]")
    print(f"       action_net.weight[1,:] drift   = {str_drift:.2e}  [steer row]")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d27.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d27 model to {save_path}")
    print(f"        Run evaluate.py to compare against d26 (1883, pit preserved).")
    print(f"        If reward > 1883 AND pit_count ≥ 1: SUCCESS (timing improved).")
    print(f"        If reward ≈ 1883 AND pit_count ≥ 1: PARTIAL (timing preserved).")


if __name__ == "__main__":
    train()
