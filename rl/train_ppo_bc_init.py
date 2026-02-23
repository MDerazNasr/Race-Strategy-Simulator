"""
PPO Training with Behavioral Cloning Weight Initialization.

STRATEGY OVERVIEW
=================
We use BC weights as a warm start for PPO. This is the "BC + RL" paradigm
used in robotics and game AI. The idea:

  Phase 1 (offline, already done): Train BC from expert demos → bc_policy_final.pt
  Phase 2 (this file): Load BC weights into PPO actor, then run RL.

WHY WARM START?
  - PPO from scratch must explore randomly to discover good behavior.
    On a racing task, random actions almost always crash immediately,
    giving the agent very little useful gradient signal early on.
  - BC warm start gives the actor a reasonable prior:
    "follow the track" is already encoded. PPO then fine-tunes this
    to maximize cumulative reward rather than just copy the expert.
  - Expected result: BC-init PPO converges in ~100k steps vs ~500k from scratch.

ARCHITECTURE ALIGNMENT
======================
We force SB3 to build the exact same hidden architecture as BCPolicy
using policy_kwargs. This makes weight transfer exact.

    BCPolicy:       Linear(6,128) → ReLU → Linear(128,128) → ReLU → Linear(128,2) → Tanh
    SB3 MlpPolicy:  policy_net = [Linear(6,128), ReLU, Linear(128,128), ReLU]
                    action_net = Linear(128, 2)
                    (SB3 applies tanh via the distribution bijector, not in the network)

HYPERPARAMETER NOTES
====================
  learning_rate=3e-4:
    Standard PPO learning rate. Adam with 3e-4 is the SB3 default and
    generally robust. BC-init may benefit from a slightly lower lr
    (e.g. 1e-4) to avoid destroying BC weights early in training.
    We use 3e-4 here as the standard baseline.

  n_steps=2048:
    Number of environment steps to collect before each PPO update.
    Larger = lower variance gradient estimates, but slower updates.
    2048 is the SB3 default for continuous control.

  batch_size=256:
    PPO shuffles the rollout buffer and takes minibatch gradient steps.
    With n_steps=2048, we get 2048/256 = 8 minibatches per update.

  n_epochs=10:
    How many passes over the rollout buffer per update.
    PPO's clipping mechanism makes multiple passes safe (unlike vanilla PG).
    10 is the SB3 default.

  gamma=0.99:
    Discount factor. At dt=0.1s, γ=0.99 means rewards 100 steps away
    (10 seconds of real time) still contribute ~0.37 of their value.
    This is appropriate for a continuous racing task.

  gae_lambda=0.95:
    Generalized Advantage Estimation parameter.
    λ=1.0 → Monte Carlo advantage (high variance, unbiased)
    λ=0.0 → TD(0) advantage (low variance, high bias)
    λ=0.95 → trade-off, standard for continuous control.

    GAE formula:
      A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
      where δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)

  clip_range=0.2:
    PPO clips the probability ratio r_t = π(a|s)/π_old(a|s) to [0.8, 1.2].
    This prevents large policy updates that destabilize learning.
    Standard value; no reason to change unless training is unstable.

  ent_coef=0.0:
    Entropy bonus coefficient. Adding entropy encourages exploration.
    We set 0.0 because BC already provides a good initialization and
    we don't want to add noise on top of a good prior.
    If PPO gets stuck in local optima, try ent_coef=0.01.
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

from rl.bc_init_policy import load_bc_weights_into_ppo, verify_transfer
from rl.make_env import make_env


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # ── Build vectorized environment ─────────────────────────────────────────
    # DummyVecEnv wraps the gym env in a vectorized interface that SB3 expects.
    # "Dummy" = single environment, no parallelism. For faster training you
    # could use SubprocVecEnv([make_env]*N) to run N envs in parallel.
    env = DummyVecEnv([make_env])

    # ── Instantiate PPO with architecture matching BCPolicy ──────────────────
    # policy_kwargs forces the actor (pi) and critic (vf) to both use
    # [128, 128] hidden layers. This is required so BC weight shapes match.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,       # No entropy bonus — BC warm start is already explorative enough
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device=device,
    )

    # ── Configure TensorBoard logging ────────────────────────────────────────
    # Run: tensorboard --logdir runs/  to compare bc_init vs scratch live.
    model.set_logger(configure("runs/ppo_bc_init", ["stdout", "tensorboard"]))

    # ── Transfer BC weights into PPO actor ───────────────────────────────────
    # This copies all 3 linear layers: trunk (2 layers) + output head.
    # After this call the actor is a trained BC policy, not a random policy.
    bc_path = str(project_root / "bc" / "bc_policy_final.pt")
    load_bc_weights_into_ppo(model, bc_path, device)

    # ── Verify the transfer was exact ────────────────────────────────────────
    # Prints max absolute weight differences. All should be ~0.0.
    verify_transfer(model, bc_path, device)

    # ── RL training ──────────────────────────────────────────────────────────
    # PPO will now fine-tune the BC actor using environment reward.
    # The critic starts from random weights (it wasn't pre-trained) and will
    # quickly learn to estimate values because the actor is already reasonable.
    print("\n[Train] Starting PPO fine-tuning from BC initialization...")
    model.learn(total_timesteps=300_000)

    # ── Save final model ─────────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_bc_init.zip")
    model.save(save_path)
    print(f"[Train] Saved PPO (BC init) to {save_path}")


if __name__ == "__main__":
    train()
