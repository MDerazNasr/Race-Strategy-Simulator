"""
BC Weight Transfer Utility for SB3 PPO.

WHY NOT A CUSTOM ActorCriticPolicy SUBCLASS?
============================================
The original approach here tried to subclass ActorCriticPolicy. There were
two bugs that made it non-functional:

  Bug 1 (Python scoping):
    `def forward(...)` was INDENTED inside `__init__`, making it a local
    function, not a class method. Python silently uses the parent class
    `forward()` instead. No error is thrown — it just silently ignores
    your override.

  Bug 2 (log_prob = 0 breaks PPO):
    PPO's policy gradient uses the probability ratio:
        r_t = π_θ(a|s) / π_θ_old(a|s)
    Both are computed via log_prob. If log_prob = 0 for all actions,
    then π(a|s) = e^0 = 1 and r_t = 1/1 = 1 always.
    The clipped objective has zero gradient → the actor NEVER updates.
    PPO degenerates to fitting the value function only.

THE CORRECT APPROACH (production pattern):
  1. Use `policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[128,128]))`
     to tell SB3 to build an actor network of the same size as your BC MLP.
  2. After building the PPO model, manually copy BC weights with torch.no_grad().
  3. SB3 handles all distribution math (Gaussian sampling, log_probs, entropy)
     correctly via its built-in DiagGaussianDistribution.

This module provides `load_bc_weights_into_ppo()` as a clean, reusable utility.
It is used by train_ppo_bc_init.py.

SB3 ACTOR ARCHITECTURE (with net_arch=dict(pi=[128,128], vf=[128,128])):
=========================================================================
    obs (shape [B, obs_dim])   ← obs_dim = 11 after Part B+A (was 6)
        │
        ▼
    FlattenExtractor         ← identity for flat obs, just ensures shape (B, obs_dim)
        │
        ▼
    mlp_extractor.policy_net ← shared trunk for actor
        Linear(obs_dim, 128)
        ReLU()
        Linear(128, 128)
        ReLU()
        │
        ▼
    action_net               ← actor head: Linear(128, 2)
        │
        ▼
    DiagGaussianDistribution ← squashes with tanh, computes log_prob, entropy
        (samples action, computes log_prob for PPO update)

BC ARCHITECTURE (BCPolicy in bc/train_bc.py):
=============================================
    obs (shape [B, obs_dim])   ← obs_dim inferred from saved weights
        │
        ▼
    net[0]: Linear(obs_dim, 128)
    net[1]: ReLU()
    net[2]: Linear(128, 128)
    net[3]: ReLU()
    net[4]: Linear(128, 2)    ← weight transfer target: action_net
    net[5]: Tanh()            ← BC applied tanh in-network; SB3 applies it
                                 via the distribution bijector. Same math.

WEIGHT MAPPING:
    BC net[0]  → SB3 policy_net[0]  (first hidden layer)
    BC net[2]  → SB3 policy_net[2]  (second hidden layer)
    BC net[4]  → SB3 action_net     (output layer)
"""

import sys
from pathlib import Path

import torch

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bc.train_bc import BCPolicy


def load_bc_weights_into_ppo(ppo_model, bc_path: str, device: str) -> None:
    """
    Copy trained BC weights into the actor network of an SB3 PPO model.

    The PPO model must have been created with:
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    so the actor architecture matches BCPolicy exactly.

    Args:
        ppo_model:  An instantiated stable_baselines3.PPO model.
        bc_path:    Path to the saved BC weights file (e.g. 'bc/bc_policy_final.pt').
        device:     PyTorch device string ('cpu' or 'cuda').

    Why this matters:
        Without this, PPO starts with random actor weights and must discover
        stable driving from scratch — this takes ~500k steps.
        With BC initialization, the actor already knows how to follow the track,
        so PPO only needs to optimize (not discover) good behavior.
        In robotics: this is called "warm starting from demonstrations".
    """

    # ── Load the trained BC policy ──────────────────────────────────────────
    # Auto-detect state_dim from saved weights so this works for any obs size.
    # 'net.0.weight' has shape (hidden_dim, state_dim) — the input dimension
    # is in axis 1.  This means no hardcoded "6" here; 11D obs just works.
    ckpt = torch.load(bc_path, map_location=device, weights_only=True)
    state_dim = ckpt["net.0.weight"].shape[1]   # e.g. 11 after Part B+A
    action_dim = ckpt["net.4.weight"].shape[0]  # always 2

    bc = BCPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    bc.load_state_dict(ckpt)
    bc.eval()

    # Extract only the Linear layers from BC's Sequential net.
    # bc.net = [Linear, ReLU, Linear, ReLU, Linear, Tanh]
    # We want the three Linear layers (indices 0, 2, 4 in bc.net).
    bc_linear_layers = [m for m in bc.net if isinstance(m, torch.nn.Linear)]
    # bc_linear_layers[0]: Linear(6, 128)   — first hidden
    # bc_linear_layers[1]: Linear(128, 128) — second hidden
    # bc_linear_layers[2]: Linear(128, 2)   — output head

    # ── Get SB3 PPO actor network components ────────────────────────────────
    # policy_net is the shared trunk: [Linear(6,128), ReLU, Linear(128,128), ReLU]
    policy_net = ppo_model.policy.mlp_extractor.policy_net
    # action_net is the output head: Linear(128, 2)
    action_net = ppo_model.policy.action_net

    # ── Copy weights with no gradient tracking ──────────────────────────────
    # torch.no_grad() is required: we are directly overwriting parameter tensors,
    # not doing a forward pass. Without it, PyTorch autograd gets confused.
    with torch.no_grad():
        # First hidden layer: BC[0] → PPO policy_net[0]
        policy_net[0].weight.copy_(bc_linear_layers[0].weight)
        policy_net[0].bias.copy_(bc_linear_layers[0].bias)

        # Second hidden layer: BC[1] → PPO policy_net[2]
        # (Index 2 because policy_net = [Linear, ReLU, Linear, ReLU])
        policy_net[2].weight.copy_(bc_linear_layers[1].weight)
        policy_net[2].bias.copy_(bc_linear_layers[1].bias)

        # Output layer: BC[2] → PPO action_net
        # This was missing in the original code — the actor head was random.
        action_net.weight.copy_(bc_linear_layers[2].weight)
        action_net.bias.copy_(bc_linear_layers[2].bias)

    print(
        f"[BC Init] Transferred weights from '{bc_path}' into PPO actor.\n"
        f"  policy_net[0]: {policy_net[0].weight.shape} ✓\n"
        f"  policy_net[2]: {policy_net[2].weight.shape} ✓\n"
        f"  action_net:    {action_net.weight.shape} ✓"
    )


def extend_obs_dim(model, old_dim: int, new_dim: int) -> None:
    """
    Extend an SB3 PPO model's input from old_dim to new_dim observations
    by zero-padding the new input weight columns.

    WHY THIS IS NEEDED (Week 5 — tyre degradation):
    ================================================
    All previous policies (v2, multi_lap, etc.) were trained with an 11D
    observation vector.  The tyre degradation env adds a 12th dimension
    (tyre_life).  We cannot directly load a 11D-input policy into a 12D env
    because the first Linear layer's weight matrix has the wrong shape:

        Old:  Linear(11, 128) → weight shape (128, 11)
        New:  Linear(12, 128) → weight shape (128, 12)

    SB3's PPO.load() would raise a shape mismatch error.

    HOW IT WORKS:
    =============
    For each network (actor trunk, critic trunk):
      1. Copy the existing (128, 11) weight into a new (128, 12) tensor.
      2. Set the new column (column index 11) to zero.
      3. Replace the parameter in-place.

    Zero-initialising the new column means:
      - The policy behaves IDENTICALLY to v2 on the first step
        (the new obs dimension contributes zero to every neuron).
      - Over training, the new column's weights learn to respond to tyre_life.
      - This is the principled way to extend NLP embedding matrices too —
        same idea as adding a new token to a language model's vocabulary.

    The bias vector is NOT touched — it's shape (128,), independent of input dim.

    After calling this function, update model.observation_space and
    model.policy.observation_space to match the new 12D space before
    calling model.set_env().

    Args:
        model:    A loaded SB3 PPO model (11D obs, from ppo_curriculum_v2).
        old_dim:  Current input dimension (11).
        new_dim:  Target input dimension (12).
    """
    import torch
    from gymnasium import spaces
    import numpy as np

    assert new_dim > old_dim, "new_dim must be larger than old_dim"

    with torch.no_grad():
        for net in [
            model.policy.mlp_extractor.policy_net,   # actor trunk
            model.policy.mlp_extractor.value_net,    # critic trunk
        ]:
            first_layer = net[0]  # Linear(old_dim, 128)
            old_w = first_layer.weight.data  # shape (128, old_dim)

            # Build new weight: copy old columns, zero-pad the rest.
            new_w = torch.zeros(
                old_w.shape[0], new_dim,
                dtype=old_w.dtype,
                device=old_w.device,
            )
            new_w[:, :old_dim] = old_w   # copy existing knowledge
            # new_w[:, old_dim:] is already 0 — new dims start silent

            # Replace the parameter and update in_features for correctness.
            first_layer.weight = torch.nn.Parameter(new_w)
            first_layer.in_features = new_dim

    # Update the observation space stored on the model so SB3 doesn't
    # reject the new env when set_env() is called.
    old_space = model.observation_space
    new_high = np.append(old_space.high, np.ones(new_dim - old_dim)).astype(np.float32)
    new_low  = np.append(old_space.low,  np.zeros(new_dim - old_dim)).astype(np.float32)
    new_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    model.observation_space         = new_space
    model.policy.observation_space  = new_space

    print(
        f"[extend_obs_dim] Expanded input {old_dim}D → {new_dim}D.\n"
        f"  policy_net[0]: weight shape now {model.policy.mlp_extractor.policy_net[0].weight.shape}\n"
        f"  value_net[0]:  weight shape now {model.policy.mlp_extractor.value_net[0].weight.shape}\n"
        f"  New obs space: {new_space}"
    )


def verify_transfer(ppo_model, bc_path: str, device: str) -> None:
    """
    Sanity check: verifies the weight transfer was exact.
    Prints max absolute difference between BC and PPO actor weights.
    All values should be ~0.0 if transfer succeeded.

    Use after calling load_bc_weights_into_ppo() to confirm correctness.
    """
    ckpt = torch.load(bc_path, map_location=device, weights_only=True)
    state_dim  = ckpt["net.0.weight"].shape[1]
    action_dim = ckpt["net.4.weight"].shape[0]
    bc = BCPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    bc.load_state_dict(ckpt)
    bc.eval()

    bc_layers = [m for m in bc.net if isinstance(m, torch.nn.Linear)]
    policy_net = ppo_model.policy.mlp_extractor.policy_net
    action_net = ppo_model.policy.action_net

    diffs = [
        ("policy_net[0] weight", (policy_net[0].weight - bc_layers[0].weight).abs().max().item()),
        ("policy_net[0] bias",   (policy_net[0].bias   - bc_layers[0].bias  ).abs().max().item()),
        ("policy_net[2] weight", (policy_net[2].weight - bc_layers[1].weight).abs().max().item()),
        ("policy_net[2] bias",   (policy_net[2].bias   - bc_layers[1].bias  ).abs().max().item()),
        ("action_net weight",    (action_net.weight    - bc_layers[2].weight).abs().max().item()),
        ("action_net bias",      (action_net.bias      - bc_layers[2].bias  ).abs().max().item()),
    ]

    print("[BC Init] Weight transfer verification:")
    for name, diff in diffs:
        status = "✓" if diff < 1e-6 else "✗ MISMATCH"
        print(f"  {name}: max_diff = {diff:.2e}  {status}")
