"""
PPO Pit Strategy v4 D31 — BC Pre-training of Pit Row (Week 5).

WHY D31?
=========
D26–D30 all failed because the gradient consistently pushes pit_signal negative:
  - D26–D29: forced pits fire regardless of agent's pit_signal. PPO credits positive
    advantage to the agent's chosen action (usually pit_signal < 0). Bias → -1.22.
  - D30: voluntary_pit_reward (+300 when pit_signal>0 AND tyre_life<0.60) provides
    the right signal, but early over-exploration swamps it.
    With pit_std=2.13 and near-zero bias, agent pits ~50% of steps on FRESH tyres
    → -200 penalties dominate → gradient pushes pit bias more negative before the
    +300 bonus zone is ever reached consistently.

THE D31 FIX — BC PRE-TRAINING OF PIT ROW:
  1. Load d21 (frozen features, good driving, near-zero pit bias)
  2. Collect states by driving d21 WITHOUT pitting (pit_signal=-1.0)
  3. Create supervised dataset from those states:
       target = +2.0  when tyre_life < 0.55  (worn → should pit)
       target = -2.0  when tyre_life > 0.65  (fresh → should NOT pit)
  4. Run gradient descent on ONLY action_net.weight[2,:] + bias[2]
     (frozen features × trainable pit row = state-conditional linear classifier)
  5. After BC: pit_signal > 0 at worn states, < 0 at fresh states
  6. Re-initialize log_std[2] to d21's original value (restore exploration)
  7. Apply three-layer freeze, then run PPO with voluntary_pit_reward=True

WHY BC PRE-TRAINING WORKS:
  After step 4, the pit row is initialized to:
    - Output pit_signal ≈ +2.0 when the feature vector encodes tyre_life < 0.55
    - Output pit_signal ≈ -2.0 when the feature vector encodes tyre_life > 0.65
  This is achievable because tyre_life is in obs[11] → features distinguish states.

  During PPO fine-tuning:
    - At worn-tyre states: pit fires → +300 voluntary bonus → net +100
      → POSITIVE advantage → gradient REINFORCES pit_signal > 0 at worn states ✓
    - At fresh-tyre states: pit_signal < 0 → pit doesn't fire → no -200 penalty
      → Neutral/positive gradient (no fresh-tyre penalty to fight) ✓
  The early over-exploration problem is solved: the BC makes pitting STATE-CONDITIONAL
  from the first training episode.

LR BUG FIX (same as d30):
  Set BOTH model.learning_rate AND model.lr_schedule = cosine(1e-4→1e-6).
  D29 bug: only set learning_rate; SB3 uses lr_schedule for optimizer updates.

STARTING POINT: ppo_pit_v4.zip (d21, reward=1877, 7 laps, 1 pit, bias=+0.006)
SAVES TO:       rl/ppo_pit_v4_d31.zip
LOGS TO:        runs/ppo_pit_v4_d31/
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env
from rl.make_env import make_env_pit_d30       # voluntary_pit_reward=True, no forced pit
from rl.schedules import cosine_schedule


# ── BC hyperparameters ────────────────────────────────────────────────────────
BC_EPISODES          = 100    # collect 100 episodes of driving data
BC_EPOCHS            = 500    # gradient descent epochs for pit row
BC_LR                = 1e-3   # Adam learning rate for BC
BC_WORN_THRESHOLD    = 0.55   # tyre_life below this → target = +2.0 (pit!)
BC_FRESH_THRESHOLD   = 0.65   # tyre_life above this → target = -2.0 (don't pit!)
BC_TARGET_WORN       = 2.0    # target pit_signal for worn states
BC_TARGET_FRESH      = -2.0   # target pit_signal for fresh states


def collect_bc_states(model, device):
    """
    Drive 100 episodes with d21 policy (no pitting) to collect (obs, tyre_life) pairs.
    Returns: (features_tensor, tyre_lives_array) where features are from the frozen
    mlp_extractor.policy_net. The pit row will be trained to separate worn vs fresh.
    """
    print(f"\n[BC] Collecting {BC_EPISODES} episodes of driving data (no pitting)...")
    env = F1Env(tyre_degradation=True, pit_stops=True)

    all_obs = []
    all_tyre_lives = []

    for ep in range(BC_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=False)
            # Suppress pit signal — collect state distribution only
            action_no_pit = np.array([action[0], action[1], -1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_no_pit)
            all_obs.append(obs.copy())
            all_tyre_lives.append(info["tyre_life"])
            done = terminated or truncated

    env.close()

    n = len(all_obs)
    print(f"[BC] Collected {n} (obs, tyre_life) pairs from {BC_EPISODES} episodes.")

    # Extract frozen features
    obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32).to(device)
    with torch.no_grad():
        features = model.policy.mlp_extractor.policy_net(obs_tensor).cpu()

    tyre_lives = np.array(all_tyre_lives)

    worn_count  = (tyre_lives < BC_WORN_THRESHOLD).sum()
    fresh_count = (tyre_lives > BC_FRESH_THRESHOLD).sum()
    neutral_count = n - worn_count - fresh_count
    print(f"[BC] State distribution:")
    print(f"       worn  (tl < {BC_WORN_THRESHOLD}): {worn_count:,}  → target = +{BC_TARGET_WORN}")
    print(f"       neutral ({BC_WORN_THRESHOLD}–{BC_FRESH_THRESHOLD}): {neutral_count:,}  → excluded from BC")
    print(f"       fresh (tl > {BC_FRESH_THRESHOLD}): {fresh_count:,}  → target = {BC_TARGET_FRESH}")

    return features, tyre_lives


def bc_pretrain_pit_row(model, device):
    """
    BC pre-train action_net.weight[2,:] and bias[2] to be state-conditional:
      tyre_life < BC_WORN_THRESHOLD  → pit_signal ≈ +2.0 (pit!)
      tyre_life > BC_FRESH_THRESHOLD → pit_signal ≈ -2.0 (don't pit!)
    Only trains these 129 parameters; all other weights are unchanged.
    """
    features, tyre_lives = collect_bc_states(model, device)

    # Create supervised targets
    worn_mask    = tyre_lives < BC_WORN_THRESHOLD
    fresh_mask   = tyre_lives > BC_FRESH_THRESHOLD
    train_mask   = worn_mask | fresh_mask

    targets = np.zeros(len(tyre_lives), dtype=np.float32)
    targets[worn_mask]  =  BC_TARGET_WORN
    targets[fresh_mask] =  BC_TARGET_FRESH

    features_train = features[train_mask]
    targets_train  = torch.tensor(targets[train_mask]).unsqueeze(1)

    print(f"\n[BC] Training pit row on {train_mask.sum()} samples "
          f"({worn_mask.sum()} worn, {fresh_mask.sum()} fresh). "
          f"Excluded {(~train_mask).sum()} neutral samples.")

    # Extract pit row params as a standalone linear module
    pit_w = model.policy.action_net.weight[2:3, :].detach().clone().requires_grad_(True)
    pit_b = model.policy.action_net.bias[2:3].detach().clone().requires_grad_(True)
    optimizer_bc = torch.optim.Adam([pit_w, pit_b], lr=BC_LR)

    best_loss = float("inf")
    for epoch in range(BC_EPOCHS):
        output = features_train @ pit_w.T + pit_b    # (N, 1)
        loss = F.mse_loss(output, targets_train)
        optimizer_bc.zero_grad()
        loss.backward()
        optimizer_bc.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"[BC] Epoch {epoch+1:>4}/{BC_EPOCHS}: loss={loss.item():.4f}  "
                  f"pit_bias={pit_b.item():+.4f}")

    # Copy BC weights back to model
    with torch.no_grad():
        model.policy.action_net.weight[2, :] = pit_w[0].to(device)
        model.policy.action_net.bias[2]      = pit_b[0].to(device)

    final_bias = model.policy.action_net.bias[2].item()
    final_w_mean = model.policy.action_net.weight[2, :].abs().mean().item()
    print(f"\n[BC] Pre-training complete.")
    print(f"     Final BC loss:  {best_loss:.4f}")
    print(f"     Pit bias after BC: {final_bias:+.4f}")
    print(f"     Pit weight abs_mean after BC: {final_w_mean:.6f}")

    # Quick sanity check on a few states
    with torch.no_grad():
        worn_sample  = features[worn_mask][:10]    if worn_mask.any()  else None
        fresh_sample = features[fresh_mask][:10]   if fresh_mask.any() else None

        if worn_sample is not None:
            worn_outputs = (worn_sample @ pit_w.T + pit_b).squeeze().numpy()
            print(f"     Pit_signal on 10 worn states (target={BC_TARGET_WORN:+.1f}):")
            print(f"       {worn_outputs.round(2)}")
            print(f"       Mean: {worn_outputs.mean():+.3f}  (should be ≈ +{BC_TARGET_WORN})")

        if fresh_sample is not None:
            fresh_outputs = (fresh_sample @ pit_w.T + pit_b).squeeze().numpy()
            print(f"     Pit_signal on 10 fresh states (target={BC_TARGET_FRESH:+.1f}):")
            print(f"       {fresh_outputs.round(2)}")
            print(f"       Mean: {fresh_outputs.mean():+.3f}  (should be ≈ {BC_TARGET_FRESH})")

    return final_bias


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
    print(f"[Train] Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"[Train] Checkpoint loaded.")

    # Pre-BC diagnostics
    pit_bias_d21    = model.policy.action_net.bias[2].item()
    pit_w_d21       = model.policy.action_net.weight[2, :].abs().mean().item()
    log_std_d21     = model.policy.log_std[2].item()
    thr_w_before    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_before    = model.policy.action_net.weight[1, :].abs().mean().item()
    print(f"\n[Diag] D21 pre-BC pit row:")
    print(f"       pit_bias    = {pit_bias_d21:+.6f}  (near-zero)")
    print(f"       pit_w_mean  = {pit_w_d21:.6f}")
    print(f"       log_std[2]  = {log_std_d21:.6f}  (std={torch.exp(torch.tensor(log_std_d21)).item():.3f})")

    # ── BC PRE-TRAINING OF PIT ROW ─────────────────────────────────────────────
    # Directly initialize pit row to be state-conditional BEFORE PPO training.
    # This guarantees the gradient goes in the right direction from the first episode.
    pit_bias_after_bc = bc_pretrain_pit_row(model, device)

    # Re-initialize log_std[2] to d21's value to restore exploration
    # (BC doesn't touch log_std, but let's confirm it's still at d21's level)
    log_std_after_bc = model.policy.log_std[2].item()
    print(f"\n[BC] log_std[2] = {log_std_after_bc:.6f} (std={torch.exp(torch.tensor(log_std_after_bc)).item():.3f})")
    print(f"     [If std < 0.5, consider resetting log_std[2] = {log_std_d21:.4f}]")
    if log_std_after_bc < 0.5:  # std < 1.65
        # Restore d21's exploration level so PPO can explore around the BC mean
        with torch.no_grad():
            model.policy.log_std[2] = torch.tensor(log_std_d21, dtype=torch.float32)
        print(f"     Reset log_std[2] = {log_std_d21:.6f} (d21's value)")

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

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[Freeze] Summary:")
    print(f"         Frozen (features):    {frozen_feat:,} params")
    print(f"         Trainable (policy):   action_net.weight[2,:] + bias[2] + log_std[2] = 130 params")
    print(f"         Trainable (value):    mlp_extractor.value_net + value_net")
    print(f"         Trainable (total):    {trainable:,} / {total:,} params")

    # ── LR schedule — both learning_rate AND lr_schedule (d29/d30 bug fix) ────
    new_schedule = cosine_schedule(initial_lr=1e-4, min_lr=1e-6)
    model.learning_rate = new_schedule
    model.lr_schedule   = new_schedule
    print(f"\n[LR] cosine(1e-4→1e-6), lr_schedule(1.0) = {new_schedule(1.0):.2e}")

    # ── Configure TensorBoard ──────────────────────────────────────────────────
    model.set_logger(configure("runs/ppo_pit_v4_d31", ["stdout", "tensorboard"]))

    # ── Run training ───────────────────────────────────────────────────────────
    print(f"\n[Train] Pit Strategy v4 D31: {TOTAL_STEPS:,} steps")
    print(f"        Starting from:       ppo_pit_v4.zip (d21) + BC pit row pre-training")
    print(f"        BC target:           pit_signal=+2.0 when tl<{BC_WORN_THRESHOLD}, -2.0 when tl>{BC_FRESH_THRESHOLD}")
    print(f"        Pit bias after BC:   {pit_bias_after_bc:+.4f} (should be near-zero or positive)")
    print(f"        Environment:         make_env_pit_d30 (voluntary_pit_reward=True)")
    print(f"        Frozen:              mlp_extractor.policy_net + throttle/steer rows")
    print(f"        Goal: pit bias stays positive; voluntary pit at tyre_life<0.60")
    print(f"              reward > 2000 on fixed-start\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,
    )

    # ── Post-training diagnostics ──────────────────────────────────────────────
    pit_w_after    = model.policy.action_net.weight[2, :].abs().mean().item()
    thr_w_after    = model.policy.action_net.weight[0, :].abs().mean().item()
    str_w_after    = model.policy.action_net.weight[1, :].abs().mean().item()
    pit_bias_after = model.policy.action_net.bias[2].item()
    pit_std_after  = model.policy.log_std[2].item()
    print(f"\n[Diag] Post-training weights:")
    print(f"       action_net.weight[0,:] = {thr_w_after:.6f}  (was {thr_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[1,:] = {str_w_after:.6f}  (was {str_w_before:.6f}) [SHOULD BE SAME]")
    print(f"       action_net.weight[2,:] = {pit_w_after:.6f}  (was {pit_w_d21:.6f}) [PIT]")
    print(f"       action_net.bias[2]     = {pit_bias_after:.6f}  (was {pit_bias_d21:.6f} d21, {pit_bias_after_bc:.6f} after BC) [PIT]")
    print(f"       log_std[2]             = {pit_std_after:.6f}  (was {log_std_d21:.6f} d21) [PIT]")

    # Freeze verification
    state_d21 = PPO.load(checkpoint_path, device=device)
    feature_drift = max(
        (p_new - p_old).abs().max().item()
        for p_new, p_old in zip(
            model.policy.mlp_extractor.policy_net.parameters(),
            state_d21.policy.mlp_extractor.policy_net.parameters()
        )
    )
    thr_drift = (model.policy.action_net.weight[0, :] - state_d21.policy.action_net.weight[0, :]).abs().max().item()
    str_drift = (model.policy.action_net.weight[1, :] - state_d21.policy.action_net.weight[1, :]).abs().max().item()

    print(f"\n[Diag] Freeze verification vs d21:")
    print(f"       features drift = {feature_drift:.2e}  [should be 0.00]")
    print(f"       throttle drift = {thr_drift:.2e}  [should be < 1e-3]")
    print(f"       steer drift    = {str_drift:.2e}  [should be < 1e-3]")

    print(f"\n[Diag] Pit bias trajectory (d21→BC→d31):")
    print(f"       d21 (load):  {pit_bias_d21:+.6f}")
    print(f"       after BC:    {pit_bias_after_bc:+.6f}")
    print(f"       d31 (final): {pit_bias_after:+.6f}")
    if pit_bias_after > 0:
        print(f"       → POSITIVE BIAS — voluntary pitting learned!")
    elif pit_bias_after > pit_bias_after_bc:
        print(f"       → Moved TOWARD positive (from BC init) — partial success")
    else:
        print(f"       → Moved NEGATIVE from BC init — PPO still fighting the pit")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_pit_v4_d31.zip")
    model.save(save_path)
    print(f"\n[Train] Saved d31 model to {save_path}")
    print(f"        Run evaluate.py to compare against d21 (1877) and d30 (1880).")
    print(f"        Target: reward > 2000 on fixed-start.")


if __name__ == "__main__":
    train()
