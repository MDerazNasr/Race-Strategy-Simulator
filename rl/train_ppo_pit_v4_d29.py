"""
PPO Pit Strategy v4 D29 — Higher Safety Net Thresholds (Week 5).

WHY D29?
=========
D29 diagnostic (diagnose_d28.py) revealed the true cause of the 1883 ceiling:

  D28 NEVER pits voluntarily on fixed-start (pit_signal = -1.0, 1354 steps).
  The 1883 comes from 7 laps with ZERO pit stops.

  Root cause 1 — STAGES_PIT_V5 thresholds too low:
    Safety net at 0.25/0.15/0.08 NEVER fires on fixed-start.
    Fixed-start crashes at tyre_life=0.35 (step 1354) before these thresholds.
    → D26/D27/D28 received NO forced pit training signal on fixed-start.

  Root cause 2 — pit_timing_reward gradient backfired:
    The +100 bonus requires tyre_life < 0.30, unreachable without a prior pit.
    Gradient pushed bias more negative (-0.817 → -0.839 → -0.929).
    WRONG direction — agent learns to delay pitting when we need it to pit sooner.

COUNTERFACTUAL EVIDENCE:
  diagnose_d28.py ran forced-pit simulations to find the optimal timing:
    threshold=0.50 (step ~1145, tl≈0.50): reward=2701, 11 laps, 2000 steps!
    threshold=0.40 (step ~1280, tl≈0.39): reward=2697, 11 laps, 2000 steps!
    threshold=0.35 (step=1354,  crash):   reward=1883,  7 laps,  no benefit
    No pit (d28 actual):                  reward=1883,  7 laps, 1354 steps

  A single pit at tyre_life ≈ 0.40–0.50 unlocks 800+ more reward!
  The crash at step 1354 is tyre-INDUCED (obs[11]=tyre_life affects the
  throttle/steer action at the critical corner). Fresh tyres (obs[11]=1.0)
  allow the agent to navigate past that corner and survive to step 2000.

THE D29 FIX — STAGES_PIT_V6:
  Thresholds raised so forced pits fire BEFORE the fixed-start crash:
    Stage 0: threshold=0.50 → fires at step ~1145 (before crash at 1354)
    Stage 1: threshold=0.45 → fires at step ~1259 (before crash at 1354)
    Stage 2: threshold=0.38 → fires at step ~1322 (before crash at 1354)

  After forced pit at tyre_life≈0.50: agent survives to step 2000 (11 laps).
  Advantage of pitting: 2701 - 1883 = +818. Enormous positive signal!
  The value function learns: Q(s_tyre050, pit) >> Q(s_tyre050, no-pit).
  The pit row receives gradient to fire at tyre_life ≈ 0.45–0.55 states.

ALSO:
  - Load d21 (not d26/d28): d21 bias≈+0.006 (demonstrated voluntary pitting!).
    D26/D27/D28 have bias=-0.817/−0.839/−0.929 — wrong direction after bad training.
  - Standard env (make_env_pit, NO pit_timing_reward): removes the misaligned
    gradient that rewarded unreachable tyre_life < 0.30.
  - Three-layer freeze: identical to d26/d27/d28. Driving preserved at d21 values.
  - reset_num_timesteps=True: LR starts at 1e-4 (d28 fix retained).

STARTING POINT: ppo_pit_v4.zip (d21, reward=1877, 7 laps, 1 pit, voluntary pitting)
SAVES TO:       rl/ppo_pit_v4_d29.zip
LOGS TO:        runs/ppo_pit_v4_d29/
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

from rl.make_env import make_env_pit          # standard env (NO pit_timing_reward)
from rl.curriculum import CurriculumCallback, STAGES_PIT_V6
from rl.schedules import cosine_schedule


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    TOTAL_STEPS = 2_000_000

    checkpoint_path = str(project_root / "rl" / "ppo_pit_v4.zip")

    # ── Prerequisite check ─────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"[Train] ppo_pit_v4.zip not found at {checkpoint_path}.\n"
            "        Run train_ppo_pit_v4.py (d21) first."
        )
    print(f"[Train] Starting from: {checkpoint_path} (d21, reward=1877, pits=1 on fixed-start)")

    # ── Build environment ──────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_pit])    # standard env, no pit_timing_reward

    # ── Load d21 checkpoint ───────────────────────────────────────────────────
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Pre-training diagnostics
    pit_w_before    = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_before    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_before    = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_before = model.policy.action_net.bias[2].item()
    thr_std_before  = model.policy.log_std[0].item()
    str_std_before  = model.policy.log_std[1].item()
    pit_std_before  = model.policy.log_std[2].item()
    print(f"\n[Diag] Pre-training weights (d21 pit row):")
    print(f"       action_net.weight[0,:] abs_mean = {thr_w_before:.6f}  (throttle)")
    print(f"       action_net.weight[1,:] abs_mean = {str_w_before:.6f}  (steer)")
    print(f"       action_net.weight[2,:] abs_mean = {pit_w_before:.6f}  (pit)")
    print(f"       action_net.bias[2]     = {pit_bias_before:.6f}  (pit bias — d21 near-zero)")
    print(f"       log_std[0] = {thr_std_before:.6f} (throttle std)")
    print(f"       log_std[1] = {str_std_before:.6f} (steer std)")
    print(f"       log_std[2] = {pit_std_before:.6f} (pit std)")

    # ── LAYER 1: Freeze mlp_extractor.policy_net ──────────────────────────────
    frozen_feat = 0
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
        frozen_feat += param.numel()
    print(f"\n[Freeze] Layer 1: mlp_extractor.policy_net ({frozen_feat:,} params frozen)")

    # ── LAYER 2: Freeze throttle/steer rows via gradient hooks ────────────────

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

    # ── LAYER 3: Freeze log_std throttle/steer dims ──────────────────────────

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
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 130 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── LR schedule ───────────────────────────────────────────────────────────
    # reset_num_timesteps=True below ensures progress starts at 1.0 (LR = 1e-4).
    model.learning_rate = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)

    # ── Configure curriculum callback ─────────────────────────────────────────
    callback = CurriculumCallback(stages=STAGES_PIT_V6, verbose=1)

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d29", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D29: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:       ppo_pit_v4.zip (d21, reward=1877, 1 pit)")
    print(f"        Starting LR:         1e-4 (cosine decay to 1e-6 — fresh start)")
    print(f"        Environment:         make_env_pit (NO pit_timing_reward)")
    print(f"        reset_num_timesteps: True")
    print(f"        Frozen (Layer 1):    mlp_extractor.policy_net ({frozen_feat:,} params)")
    print(f"        Frozen (Layer 2):    action_net rows 0,1 via hooks")
    print(f"        Frozen (Layer 3):    log_std[0,1] via hook")
    print(f"        Trainable policy:    130 params (pit row only)")
    print(f"        Curriculum:          STAGES_PIT_V6")
    print(f"          Stage 0 (threshold=0.50, ~500k steps): fires at step ~1145 on fixed-start")
    print(f"          Stage 1 (threshold=0.45, ~1M steps):   fires at step ~1259 on fixed-start")
    print(f"          Stage 2 (threshold=0.38, ~500k steps): fires at step ~1322 on fixed-start")
    print(f"        Goal: voluntary pit at tyre_life ≈ 0.40–0.50 on fixed-start")
    print(f"              reward > 2000 (target: ~2697 if pit fires before step 1354)")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,    # LR starts at 1e-4
        callback=callback,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pit_w_after   = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_after   = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after   = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_after = model.policy.action_net.bias[2].item()
    pit_std_after  = model.policy.log_std[2].item()
    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] = {thr_w_after:.6f}  (was {thr_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[1,:] = {str_w_after:.6f}  (was {str_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[2,:] = {pit_w_after:.6f}  (was {pit_w_before:.6f}) [PIT: can change]")
    print(f"       action_net.bias[2]     = {pit_bias_after:.6f}  (was {pit_bias_before:.6f}) [PIT: can change]")
    print(f"       log_std[2]             = {pit_std_after:.6f}  (was {pit_std_before:.6f}) [PIT: can change]")

    # Freeze verification vs d21
    state_d21 = PPO.load(checkpoint_path, device=device)

    feature_drift = max(
        (p_new - p_old).abs().max().item()
        for p_new, p_old in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            state_d21.policy.mlp_extractor.policy_net.parameters()
        )
    )
    thr_drift = (
        model.policy.action_net.weight[0, :] - state_d21.policy.action_net.weight[0, :]
    ).abs().max().item()
    str_drift = (
        model.policy.action_net.weight[1, :] - state_d21.policy.action_net.weight[1, :]
    ).abs().max().item()

    print(f"\n[Diag] Freeze verification vs d21 (all should be ≈ 0.0):")
    print(f"       mlp_extractor.policy_net drift = {feature_drift:.2e}  [features]")
    print(f"       action_net.weight[0,:] drift   = {thr_drift:.2e}  [throttle row]")
    print(f"       action_net.weight[1,:] drift   = {str_drift:.2e}  [steer row]")

    # Key diagnosis: did pit bias go POSITIVE?
    print(f"\n[Diag] Pit bias trajectory (d21→d29):")
    print(f"       d21 (start): {pit_bias_before:+.6f}")
    print(f"       d29 (end):   {pit_bias_after:+.6f}")
    if pit_bias_after > pit_bias_before:
        print(f"       → POSITIVE SHIFT (+{pit_bias_after - pit_bias_before:.4f}) — pit row moving toward pitting!")
    else:
        print(f"       → NEGATIVE SHIFT ({pit_bias_after - pit_bias_before:.4f}) — unexpected (investigate)")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d29.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d29 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (1877) and d28 (1883).")
    print(f"        If reward > 2000: voluntary pit fired before crash at step 1354.")
    print(f"        Target: ~2697 (forced-pit counterfactual at tyre_life<0.40).")


if __name__ == "__main__":
    train()
