"""
Learning Rate Schedules for PPO Training.

WHY DOES THE LEARNING RATE MATTER?
====================================
Every time the neural network updates its weights, it takes a "step"
in the direction that reduces the loss. The learning rate (lr) controls
how BIG that step is.

  lr too large  → takes huge steps, overshoots good solutions, training is chaotic
  lr too small  → takes tiny steps, learning is painfully slow
  lr that decays → starts bold early on, becomes cautious as it converges

Analogy: Imagine you're blindfolded trying to walk to the lowest point
of a hilly landscape. Early on you take big strides to cover ground fast.
Once you're roughly in the right valley, you switch to tiny steps so you
don't accidentally walk back uphill. That's learning rate decay.

WHY IS THIS ESPECIALLY IMPORTANT FOR PPO?
==========================================
PPO (Proximal Policy Optimization) is an on-policy algorithm.
Every gradient update uses FRESH data collected by the CURRENT policy.
This means:

  - Early training: policy is far from optimal, we want fast change
  - Late training:  policy found a good trajectory, we want STABILITY

If you keep a constant high learning rate in late training, you risk
"catastrophic forgetting" — the policy destroys good behavior with a
slightly-too-large gradient step. This is exactly what we saw in our v2
run (the reward dip between 150k-250k steps).

A decaying learning rate is the simplest fix: let the policy be bold early,
then conservative when it's close to a good solution.

WHAT IS SB3'S LEARNING RATE INTERFACE?
=======================================
Stable-Baselines3 accepts a callable (a function) as the learning_rate
argument instead of a fixed number. SB3 calls this function at every
gradient update, passing it one argument: `progress_remaining`.

    progress_remaining = 1.0  at the very start of training
    progress_remaining = 0.5  halfway through training
    progress_remaining = 0.0  at the very end of training

So your schedule function must:
    Input:  a float in [0.0, 1.0]   (1 = just started, 0 = nearly done)
    Output: the learning rate to use right now

This is a standard Python "closure" pattern — a function that returns
another function. The outer function takes the initial LR as a parameter,
and the inner function is what SB3 actually calls.
"""

import math


def linear_schedule(initial_lr: float):
    """
    Linearly decay the learning rate from initial_lr to 0.

    The learning rate at any point in training is:
        lr(t) = initial_lr * progress_remaining

    Where progress_remaining goes from 1.0 (start) to 0.0 (end).

    Visual shape:
        lr
        |\\
        | \\
        |  \\
        |   \\
        |    \\
        +---------> timesteps
        start    end

    PROS:
      - Simple and predictable
      - Guaranteed to reach very small lr at the end

    CONS:
      - Drops rate aggressively early (when you still want bold updates)
      - The "slope" is the same throughout — doesn't match how RL learns

    When to use:
      - Good baseline, works in most settings
      - When you want a simple, explainable schedule

    Interview tip:
      "I used a linear decay schedule so the agent takes bold updates
       early when the policy is far from optimal, and conservative updates
       later when we're fine-tuning near a good solution."

    Args:
        initial_lr: The learning rate at the start of training (e.g. 3e-4)

    Returns:
        A function that SB3 calls at each update, returning the current lr.
    """

    def schedule(progress_remaining: float) -> float:
        """
        Args:
            progress_remaining: float in [0.0, 1.0]
                1.0 = training just started
                0.0 = training just finished

        Returns:
            The learning rate to use right now.
        """
        # At progress=1.0 (start): lr = initial_lr * 1.0 = initial_lr
        # At progress=0.5 (half):  lr = initial_lr * 0.5
        # At progress=0.0 (end):   lr = initial_lr * 0.0 = 0
        return progress_remaining * initial_lr

    return schedule


def cosine_schedule(initial_lr: float, min_lr: float = 1e-6):
    """
    Cosine annealing schedule: decay the learning rate following a cosine curve.

    The learning rate at any point in training is:
        fraction = 1 - progress_remaining          (0 = start, 1 = end)
        lr(t) = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * fraction))

    This is the standard "cosine annealing" formula from the paper:
    "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2017)

    Let's verify the boundary values:
        At start (progress=1.0): fraction=0, cos(0)=1
          lr = min_lr + 0.5*(initial_lr - min_lr)*(1+1) = initial_lr ✓

        At end   (progress=0.0): fraction=1, cos(π)=-1
          lr = min_lr + 0.5*(initial_lr - min_lr)*(1-1) = min_lr ✓

    Visual shape (compared to linear):
        lr
        |\\______
        |        \\___
        |             \\___
        |                  \\____
        |                        \\___
        +----------------------------> timesteps
          slow drop      fast drop    very slow drop
          (still bold)   (converging) (fine-tuning)

    This S-curve means:
        - EARLY training: lr is nearly constant (full exploration speed)
        - MIDDLE training: lr drops rapidly (converging to solution)
        - LATE training:  lr is nearly constant at minimum (stable fine-tuning)

    WHY COSINE IS BETTER THAN LINEAR FOR RL:
        RL learning is not uniform — it has phases:
          Phase 1: Discovery   — agent finds basic behaviors (needs high lr)
          Phase 2: Convergence — agent refines good trajectories (needs fast decay)
          Phase 3: Fine-tuning — agent makes micro-improvements (needs tiny lr)
        Cosine naturally matches these phases. Linear does not.

    WHY NOT JUST USE A CONSTANT LR?
        With constant lr=3e-4 all the way to step 300k:
        - Late in training, a gradient step of 3e-4 is large relative to
          the small improvements being made. This creates oscillation around
          the optimum instead of settling into it.
        - We observed this as the reward dip at ~150k-250k steps in the v2 run.

    Args:
        initial_lr: Starting learning rate (e.g. 3e-4)
        min_lr:     Minimum learning rate, never goes below this (e.g. 1e-6)
                    Prevents lr from going all the way to 0, which would stop
                    learning entirely and prevent adaptation to new situations.

    Returns:
        A function that SB3 calls at each update, returning the current lr.
    """

    def schedule(progress_remaining: float) -> float:
        """
        Args:
            progress_remaining: float in [0.0, 1.0], from SB3
                1.0 = start, 0.0 = end

        Returns:
            Current learning rate following a cosine decay curve.
        """
        # fraction goes from 0 (start) to 1 (end) — opposite of progress_remaining
        # This makes the cosine formula intuitive: 0 = beginning, 1 = finished
        fraction = 1.0 - progress_remaining

        # Cosine annealing: smoothly interpolate between initial_lr and min_lr
        # math.cos(math.pi * fraction):
        #   At fraction=0: cos(0) = 1      → lr = initial_lr
        #   At fraction=0.5: cos(π/2) = 0  → lr = midpoint
        #   At fraction=1: cos(π) = -1     → lr = min_lr
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))

        # Interpolate: when cosine_factor=1 → initial_lr, when =0 → min_lr
        lr = min_lr + (initial_lr - min_lr) * cosine_factor

        return lr

    return schedule


def warmup_cosine_schedule(initial_lr: float, warmup_steps: int,
                           total_steps: int, min_lr: float = 1e-6):
    """
    Cosine decay WITH a linear warmup phase at the beginning.

    Warmup means: instead of starting at full initial_lr, we ramp up
    from a near-zero lr to initial_lr over the first `warmup_steps` steps.

    WHY USE WARMUP?
        At the very start of training, if we haven't seen much data,
        the gradient estimates are NOISY (high variance). A large lr step
        on a noisy gradient can send the policy in a completely wrong direction.

        Warmup gradually increases lr, allowing the model to "stabilize"
        before taking large gradient steps.

        Analogy: F1 cars don't go from 0 to 200 mph instantly — they warm
        up the tyres first. Similarly, the policy warms up its updates before
        committing to large changes.

    Visual shape:
        lr
        |        ____________
        |       /            \\____
        |      /                   \\___
        |     /                         \\
        |    /                            \\
        +---|warmup|------cosine decay------> timesteps

    When to use:
        - When starting from BC weights (already have a good prior)
        - To be extra conservative and prevent destroying BC initialization
        - Advanced use case — not always necessary

    Args:
        initial_lr:    Peak learning rate (e.g. 3e-4)
        warmup_steps:  How many steps to ramp up (e.g. 1000)
        total_steps:   Total training steps (e.g. 300_000)
        min_lr:        Minimum lr at the end

    Returns:
        A function that SB3 calls at each update.
        NOTE: SB3 passes progress_remaining, not step count.
        We convert: current_step = (1 - progress_remaining) * total_steps
    """

    def schedule(progress_remaining: float) -> float:
        current_step = (1.0 - progress_remaining) * total_steps

        if current_step < warmup_steps:
            # Linear warmup phase: ramp from min_lr to initial_lr
            warmup_factor = current_step / max(warmup_steps, 1)
            return min_lr + (initial_lr - min_lr) * warmup_factor
        else:
            # Cosine decay phase: after warmup
            # Re-map progress to the cosine phase only
            steps_after_warmup = current_step - warmup_steps
            cosine_steps = total_steps - warmup_steps
            fraction = steps_after_warmup / max(cosine_steps, 1)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
            return min_lr + (initial_lr - min_lr) * cosine_factor

    return schedule
