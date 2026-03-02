"""
Pit-Aware Policy for F1 Pit Strategy RL (Week 5 / d32).

WHY A CUSTOM POLICY?
====================
D26–D31 all failed because the frozen mlp_extractor features do NOT linearly
encode tyre_life for pit timing:

  BC probe (d31): worn states (tl < 0.55) → pit_signal_mean = -0.222
                  fresh states (tl > 0.65) → pit_signal_mean = -0.090
  Separation: 0.13 vs target 4.0 = only 3.25% of required separability.

The mlp_extractor was trained on throttle/steer, not pit timing. Its hidden
representations don't preserve tyre_life in a direction the linear pit row
can exploit. This is a FEATURE BOTTLENECK — no amount of pit row fine-tuning
can overcome it.

THE D32 FIX — DIRECT TYRE_LIFE CONNECTION:
  Augment the mlp_extractor output with obs[11] (tyre_life) before passing
  to action_net. The action_net input grows from 128-dim to 129-dim:

      latent_pi = [mlp_extractor.policy_net(obs) | obs[11]]  (129-dim)
      pit_signal = latent_pi[:128] @ W_pit[:128] + latent_pi[128] * W_pit[128] + b_pit
                          ^--- frozen features        ^--- DIRECT tyre_life connection

  Initialize:
    W_pit[128] = -10.0   (higher tyre_life → more negative pit signal → don't pit)
    b_pit      = +7.0    (threshold at tl ≈ 0.69, adjusted from d21 features noise)

  Result at initialization (ignoring small features noise):
    tl = 0.45 (worn):  pit_signal ≈ -10×0.45 + 7 = +2.5   P(pit>0) ≈ 88%
    tl = 0.70 (thresh): pit_signal ≈ -10×0.70 + 7 =  0.0   P(pit>0) ≈ 50%
    tl = 0.90 (fresh):  pit_signal ≈ -10×0.90 + 7 = -2.0   P(pit>0) ≈ 17%

  This is STATE-CONDITIONAL from episode 1 — no BC pre-training needed!

CLASSES:
  TyrLifeAugmentedExtractor  — MlpExtractor variant that appends obs[11] to actor output
  PitAwarePolicy             — ActorCriticPolicy variant using TyrLifeAugmentedExtractor

SAVE/LOAD COMPATIBILITY:
  SB3's PPO.save() pickles the policy_class. As long as pit_aware_policy.py is
  importable (it is — part of the project), PPO.load() can reconstruct the model.
  In evaluate.py, import PitAwarePolicy before calling PPO.load() to ensure pickle
  can find the class during deserialization.
"""

import sys
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ── Tyre-life-augmented mlp extractor ─────────────────────────────────────────

class TyrLifeAugmentedExtractor(MlpExtractor):
    """
    MlpExtractor variant that appends obs[11] (tyre_life) directly to the
    actor's latent representation before passing it to the action_net.

    Policy (actor) path:
      obs (12-dim) → policy_net (128-dim) → cat([128-dim, tyre_life]) → 129-dim
      ↑ This 129-dim tensor is passed to action_net = Linear(129, 3)

    Value (critic) path:
      obs (12-dim) → value_net (128-dim) → 128-dim   (UNCHANGED)

    Because self.latent_dim_pi is overridden to 129, ActorCriticPolicy._build()
    will automatically construct action_net = Linear(129, 3) at creation time.
    """

    TYRE_LIFE_OBS_IDX: int = 11   # obs[11] = tyre_life in the 12-dim observation

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[list, dict],
        activation_fn: type,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device=device)
        # Override latent_dim_pi: 128 → 129 (extra dim = tyre_life direct connection)
        self.latent_dim_pi = self.latent_dim_pi + 1

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Actor forward pass with tyre_life augmentation.

        Args:
            features: Raw observation tensor, shape (batch, obs_dim).
                      For F1Env with tyre_degradation=True: obs_dim = 12.
                      obs[:, 11] = tyre_life ∈ [0, 1].

        Returns:
            Augmented latent tensor, shape (batch, 129).
            = [policy_net(features) | tyre_life]
        """
        policy_latent = self.policy_net(features)                   # (batch, 128)
        tyre_life     = features[:, self.TYRE_LIFE_OBS_IDX:self.TYRE_LIFE_OBS_IDX + 1]  # (batch, 1)
        return torch.cat([policy_latent, tyre_life], dim=-1)        # (batch, 129)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Value forward pass — unchanged (no tyre_life augmentation for critic)."""
        return self.value_net(features)                             # (batch, 128)


# ── Pit-aware ActorCritic policy ───────────────────────────────────────────────

class PitAwarePolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy that uses TyrLifeAugmentedExtractor as its mlp_extractor.

    Architecture:
      Identical to standard MlpPolicy EXCEPT the actor's latent dim is 129
      instead of 128, due to the tyre_life augmentation in the extractor.
      The action_net is therefore Linear(129, 3) instead of Linear(128, 3).

    Key property:
      action_net.weight[2, 128] provides a DIRECT linear connection from
      tyre_life to pit_signal, bypassing the frozen feature bottleneck.

    Usage:
      PPO(policy=PitAwarePolicy, env=env, policy_kwargs={"net_arch": dict(pi=[128,128], vf=[128,128])})

    Initialization hint (see train_ppo_pit_v4_d32.py):
      model.policy.action_net.weight[2, 128] = -10.0   ← tyre_life → pit (negative)
      model.policy.action_net.bias[2]        = +7.0    ← threshold at tl ≈ 0.69
    """

    def _build_mlp_extractor(self) -> None:
        """Override to use TyrLifeAugmentedExtractor instead of standard MlpExtractor."""
        self.mlp_extractor = TyrLifeAugmentedExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


# ── Utility: copy d21 weights into a PitAwarePolicy model ─────────────────────

def copy_d21_weights_into_pit_aware(model_d21, model_d32) -> None:
    """
    Copy d21's policy weights into a new PitAwarePolicy model.

    D21 architecture:
      mlp_extractor.policy_net: Linear(12,128)→ReLU→Linear(128,128)→ReLU
      action_net:               Linear(128, 3)

    D32 (PitAwarePolicy) architecture:
      mlp_extractor.policy_net: IDENTICAL to d21
      action_net:               Linear(129, 3)  ← extra dim for tyre_life

    Mapping:
      mlp_extractor.policy_net  →  copy verbatim (same structure)
      mlp_extractor.value_net   →  copy verbatim (same structure)
      action_net.weight[:, :128] →  copy from d21 (rows 0,1,2 × first 128 cols)
      action_net.weight[:, 128]  →  initialized: [0, 0, -10.0] (throttle, steer, pit)
      action_net.bias[:2]        →  copy from d21 (throttle, steer)
      action_net.bias[2]         →  set to +7.0 (pit threshold at tl ≈ 0.69)
      value_net                  →  copy verbatim
      log_std                    →  copy verbatim (all 3 dims)
    """
    with torch.no_grad():
        # ── mlp_extractor.policy_net ──────────────────────────────────────────
        for p_new, p_old in zip(
            model_d32.policy.mlp_extractor.policy_net.parameters(),
            model_d21.policy.mlp_extractor.policy_net.parameters(),
        ):
            p_new.copy_(p_old)

        # ── mlp_extractor.value_net ───────────────────────────────────────────
        for p_new, p_old in zip(
            model_d32.policy.mlp_extractor.value_net.parameters(),
            model_d21.policy.mlp_extractor.value_net.parameters(),
        ):
            p_new.copy_(p_old)

        # ── action_net.weight ─────────────────────────────────────────────────
        # Copy d21's 128-dim weights into the first 128 columns of the 129-dim matrix
        model_d32.policy.action_net.weight[:, :128].copy_(model_d21.policy.action_net.weight)
        # Extra column (dim 128 = tyre_life direct connection):
        #   throttle / steer: 0.0  (don't use tyre_life for driving)
        #   pit:             -10.0 (higher tyre_life → lower pit signal → don't pit)
        model_d32.policy.action_net.weight[0, 128] = 0.0     # throttle: no tyre_life
        model_d32.policy.action_net.weight[1, 128] = 0.0     # steer:    no tyre_life
        model_d32.policy.action_net.weight[2, 128] = -10.0   # pit: direct tyre_life

        # ── action_net.bias ───────────────────────────────────────────────────
        # Copy throttle/steer bias from d21
        model_d32.policy.action_net.bias[:2].copy_(model_d21.policy.action_net.bias[:2])
        # Pit bias: set to calibrated threshold value
        #   pit_signal = (d21_features_w @ features) + (-10.0 × tyre_life) + 7.0
        #   d21 features contribution on worn states ≈ -0.22 (from BC probe in d31)
        #   At tl = 0.60: pit_signal ≈ -0.15 + (-6.0) + 7.0 = 0.85 (slightly positive)
        #   At tl = 0.69: pit_signal ≈ 0.0  (50% pit probability = effective threshold)
        #   This is close to our target zone (tl < 0.60 for voluntary_pit_reward bonus)
        model_d32.policy.action_net.bias[2] = 7.0

        # ── value_net ─────────────────────────────────────────────────────────
        for p_new, p_old in zip(
            model_d32.policy.value_net.parameters(),
            model_d21.policy.value_net.parameters(),
        ):
            p_new.copy_(p_old)

        # ── log_std ───────────────────────────────────────────────────────────
        model_d32.policy.log_std.copy_(model_d21.policy.log_std)

    print("[copy_d21_weights] Done. D21 → D32 weight transfer:")
    print(f"  action_net.weight[2, :128]: d21 pit features (kept)")
    print(f"  action_net.weight[2,  128]: -10.0  (NEW: direct tyre_life connection)")
    print(f"  action_net.bias[2]:          +7.0  (NEW: threshold at tl ≈ 0.69)")
    print(f"  action_net.weight[0:2, 128]:  0.0  (throttle/steer: no tyre_life effect)")
    print(f"  All other weights: copied from d21")

    # Sanity check: verify pit signal at key tyre_life values
    print("\n[copy_d21_weights] Pit signal sanity check (features noise excluded):")
    for tl in [0.30, 0.45, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
        pit_direct = (-10.0 * tl) + 7.0
        import math
        # P(pit > 0) from N(pit_direct, std=2.13)
        pit_std = math.exp(model_d32.policy.log_std[2].item())
        z = -pit_direct / pit_std
        p_pit = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        label = "PIT" if tl < 0.60 else "hold"
        print(f"  tl={tl:.2f}: pit_signal≈{pit_direct:+.2f}  P(pit>0)≈{p_pit:.0%}  [{label}]")
