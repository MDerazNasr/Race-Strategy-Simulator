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


def zero_pit_signal_output(ppo_model, pit_action_dim: int = 2) -> None:
    """
    Zero-initialize the pit_signal output row of the PPO action_net (d20 Fix B).

    WHY THIS IS NEEDED:
    ===================
    d18 and d19 both suffered from the same problem: the BC policy initialises
    pit_signal close to -1.0 because:
      - Standard BC MSE is dominated by the 99.9% no-pit class → BC pushes
        all action outputs toward the mean of the training data ≈ -1.0.
      - Even with pit_class_weight=1000 (d20 Fix A), the BC hidden layers
        learn to associate "driving inputs" with "no pit output" — this is
        correct for most of the episode, but the output head weight for
        pit_signal starts biased toward the majority class direction.

    THE FIX:
      After transferring BC weights, zero the pit_signal row of action_net.
      This decouples pit_signal from the BC prior entirely:
        - Hidden layers [0] and [2]: PRESERVED from BC → good driving features
        - Throttle/steer rows of action_net: PRESERVED from BC → good driving
        - Pit_signal row of action_net: ZEROED → starts at N(0, σ), P(pit)=0.5

      With the zeroed pit row:
        - At initialisation, pit_signal ~ Gaussian(0, σ) where σ = exp(log_std_init)
        - P(pit_signal > 0) = 0.5 → MAXIMUM pit exploration
        - The BC driving knowledge is preserved in throttle/steer
        - The value function (bootstrapped by forced pits) teaches the agent
          when to set pit_signal > 0 vs < 0

      This is analogous to pre-training a language model and fine-tuning only
      the task-specific classification head from scratch while keeping the
      encoder frozen (except we don't freeze here — PPO updates all weights).

    CALL ORDER:
      1. load_bc_weights_into_ppo()  → transfers ALL weights (including pit row)
      2. verify_transfer()           → confirms transfer was correct
      3. zero_pit_signal_output()    → zeros pit row (AFTER verification)

    Args:
        ppo_model:     An instantiated SB3 PPO model with 3D action space.
        pit_action_dim: Index of pit_signal in the 3D action vector. Default=2.
    """
    with torch.no_grad():
        ppo_model.policy.action_net.weight.data[pit_action_dim].zero_()
        ppo_model.policy.action_net.bias.data[pit_action_dim] = 0.0

    log_std = ppo_model.policy.log_std.data[pit_action_dim].item()
    std = torch.exp(torch.tensor(log_std)).item()
    print(f"[BC Init] Zero-initialized pit_signal output (action dim {pit_action_dim}).")
    print(f"          pit_signal now: Gaussian(mean=0, std={std:.3f}) → P(pit>0) = 0.500")
    print(f"          Throttle/steer outputs: unchanged from BC transfer.")


def load_ppo_tyre_into_ppo_pit(ppo_pit_model, ppo_tyre_path: str, bc_pit_path: str, device: str) -> None:
    """
    Initialize a 3D-action pit-stop PPO from ppo_tyre weights + BC pit row (d21).

    WHY THIS IS NEEDED (d21 design):
    =================================
    d18-d20 all started from scratch (or BC only).  ppo_tyre is a strong
    prior: it has been trained for 5M+ steps on tyre-degradation driving and
    survives ~1500+ steps per episode.  This matters because:

      1. ppo_tyre's value function already knows driving quality at each state.
         Starting from this value function means Stage 0 forced pits (at
         tyre_life=0.35) give MEANINGFUL gradient — the value function has
         a calibrated baseline to update from.

      2. ppo_tyre's actor policy knows how to drive fast without crashing.
         The agent will reach step 929 (tyre_life=0.35) in every episode.
         The forced pit fires at the RIGHT state, every time.

      3. Starting from BC alone (d18-d20): ep_len collapsed to 30-50 steps
         during early training because the BC policy with pit obs was weaker.
         Episodes never reached the forced pit (interval=500) or the correct
         tyre state (threshold=0.35, ~step 929).

    THE WEIGHT TRANSFER:
    ====================
    ppo_tyre has a 2D action space [throttle, steer].
    ppo_pit has a 3D action space [throttle, steer, pit_signal].

    Hidden layers (12D obs → 128 → 128):
      ppo_tyre.policy_net → ppo_pit.policy_net  (EXACT COPY)
      ppo_tyre.value_net  → ppo_pit.value_net   (EXACT COPY)

    Action head:
      ppo_tyre.action_net[0] (throttle) → ppo_pit.action_net[0]  (EXACT COPY)
      ppo_tyre.action_net[1] (steer)    → ppo_pit.action_net[1]  (EXACT COPY)
      bc_policy_pit_v3.net[4][2] (pit)  → ppo_pit.action_net[2]  (FROM BC)

    Log standard deviations:
      ppo_tyre.log_std[0,1] → ppo_pit.log_std[0,1]  (driving std preserved)
      ppo_pit.log_std[2]: left at its random init (SB3 default ~0.0 → std=1.0)

    No zero-init of pit row (d20 Fix B removed — it caused catastrophic failure).
    The BC-initialized pit row starts near the BC prior, which is better than
    random because the BC policy was trained with pit_class_weight=1000.

    Args:
        ppo_pit_model:  Freshly built SB3 PPO with 3D action space.
        ppo_tyre_path:  Path to the saved ppo_tyre model (e.g. 'rl/ppo_tyre.zip').
        bc_pit_path:    Path to BC pit policy (e.g. 'bc/bc_policy_pit_v3.pt').
        device:         PyTorch device string ('cpu' or 'cuda').
    """
    from stable_baselines3 import PPO

    # ── Load ppo_tyre ─────────────────────────────────────────────────────
    print(f"[BC Init] Loading ppo_tyre from '{ppo_tyre_path}'...")
    ppo_tyre = PPO.load(ppo_tyre_path, device=device)

    tyre_policy_net = ppo_tyre.policy.mlp_extractor.policy_net
    tyre_value_net  = ppo_tyre.policy.mlp_extractor.value_net
    tyre_action_net = ppo_tyre.policy.action_net   # Linear(128, 2)
    tyre_log_std    = ppo_tyre.policy.log_std       # shape (2,)

    # ── Load BC pit policy (for pit_signal row) ───────────────────────────
    ckpt = torch.load(bc_pit_path, map_location=device, weights_only=True)
    # bc_policy_pit_v3 has action_dim=3: rows [0]=throttle, [1]=steer, [2]=pit
    bc_pit_weight = ckpt["net.4.weight"]  # shape (3, 128)
    bc_pit_bias   = ckpt["net.4.bias"]    # shape (3,)

    # ── Get ppo_pit network components ────────────────────────────────────
    pit_policy_net = ppo_pit_model.policy.mlp_extractor.policy_net
    pit_value_net  = ppo_pit_model.policy.mlp_extractor.value_net
    pit_action_net = ppo_pit_model.policy.action_net   # Linear(128, 3)
    pit_log_std    = ppo_pit_model.policy.log_std       # shape (3,)

    with torch.no_grad():
        # ── Transfer hidden layers (driving knowledge) ────────────────────
        pit_policy_net[0].weight.copy_(tyre_policy_net[0].weight)
        pit_policy_net[0].bias.copy_(tyre_policy_net[0].bias)
        pit_policy_net[2].weight.copy_(tyre_policy_net[2].weight)
        pit_policy_net[2].bias.copy_(tyre_policy_net[2].bias)

        pit_value_net[0].weight.copy_(tyre_value_net[0].weight)
        pit_value_net[0].bias.copy_(tyre_value_net[0].bias)
        pit_value_net[2].weight.copy_(tyre_value_net[2].weight)
        pit_value_net[2].bias.copy_(tyre_value_net[2].bias)

        # ── Transfer throttle/steer rows from ppo_tyre (rows 0,1) ─────────
        # ppo_tyre.action_net is Linear(128, 2): rows [0]=throttle, [1]=steer
        pit_action_net.weight.data[0].copy_(tyre_action_net.weight.data[0])
        pit_action_net.bias.data[0] = tyre_action_net.bias.data[0].clone()
        pit_action_net.weight.data[1].copy_(tyre_action_net.weight.data[1])
        pit_action_net.bias.data[1] = tyre_action_net.bias.data[1].clone()

        # ── Transfer pit row from BC policy (row 2) ───────────────────────
        # bc_policy_pit_v3 row 2 was trained with pit_class_weight=1000.
        # This is a better prior than random init — it at least knows
        # tyre_life < 0.3 correlates with pit_signal > 0.
        pit_action_net.weight.data[2].copy_(bc_pit_weight[2])
        pit_action_net.bias.data[2] = bc_pit_bias[2].clone()

        # ── Transfer log_std for driving dims (0,1) ───────────────────────
        # dim 2 (pit) stays at SB3's random init (~0.0 → std=1.0)
        pit_log_std.data[0] = tyre_log_std.data[0].clone()
        pit_log_std.data[1] = tyre_log_std.data[1].clone()

    # Also transfer the value head
    tyre_vf = ppo_tyre.policy.value_net
    pit_vf  = ppo_pit_model.policy.value_net
    with torch.no_grad():
        pit_vf.weight.copy_(tyre_vf.weight)
        pit_vf.bias.copy_(tyre_vf.bias)

    print(
        f"[BC Init] Transferred ppo_tyre + BC pit weights into ppo_pit actor.\n"
        f"  policy_net[0]: {pit_policy_net[0].weight.shape} ← ppo_tyre ✓\n"
        f"  policy_net[2]: {pit_policy_net[2].weight.shape} ← ppo_tyre ✓\n"
        f"  value_net:     transferred ✓\n"
        f"  value_head:    transferred ✓\n"
        f"  action_net[0] (throttle): ← ppo_tyre ✓\n"
        f"  action_net[1] (steer):    ← ppo_tyre ✓\n"
        f"  action_net[2] (pit):      ← bc_policy_pit_v3 (weighted, 1000x) ✓\n"
        f"  log_std[0,1] (driving):   ← ppo_tyre ✓\n"
        f"  log_std[2] (pit):         random init (SB3 default) — no zero-init"
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
