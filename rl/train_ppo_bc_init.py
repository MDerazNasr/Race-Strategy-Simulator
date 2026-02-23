"""
PPO Training with BC Initialization + Stability Improvements.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS PPO? (START HERE IF YOU KNOW NOTHING)
═══════════════════════════════════════════════════════════════════════════════

PPO = Proximal Policy Optimization.

"Policy" means: a function that maps observations → actions.
    f(state) → action
    e.g. "I see the car is 0.5m left of center, going 15 m/s, facing 5° off
          track direction → apply throttle=0.9, steer=-0.1"

"Optimization" means: improve the policy over time so it earns more reward.

"Proximal" (= nearby) means: we update the policy carefully, not too far from
    the previous version. This is PPO's key innovation — it prevents large,
    destructive updates that could erase everything the policy learned.

THE CORE LOOP:
    1. Run the current policy in the environment → collect (state, action, reward) data
    2. Compute how much better/worse each action was than expected → advantage
    3. Update the neural network weights to make good actions more likely
    4. Repeat from step 1 with the new policy

WHY PPO OVER OTHER RL ALGORITHMS?
    - Stable: clipping prevents catastrophic updates
    - Sample efficient for on-policy methods
    - Widely used in robotics (OpenAI Five, Boston Dynamics, etc.)
    - SB3's default for continuous control → well-tested

═══════════════════════════════════════════════════════════════════════════════
THE BC WARM START STRATEGY
═══════════════════════════════════════════════════════════════════════════════

Without warm start: PPO explores randomly → almost always crashes → tiny
    reward signal → slow, difficult convergence. In our v2 run, PPO from
    scratch achieved reward = -113 after 300k steps. It never escaped the
    "idle slowly to avoid crashing" local optimum.

With BC warm start: the policy ALREADY knows how to follow a track before
    RL even begins. PPO starts from reward = -15 (vs -38 for scratch) and
    reaches +668 in 300k steps. The improvement is ~17x.

In industry: this is called "learning from demonstrations" or "offline-to-online RL."
    DeepMind uses this for robotic manipulation. Tesla uses it for FSD.
    Starting from human demonstrations (or BC) cuts training time dramatically.

═══════════════════════════════════════════════════════════════════════════════
VERSION HISTORY AND WHAT WE FIXED
═══════════════════════════════════════════════════════════════════════════════

v1 (train_ppo_bc_init.py original):
    Bug: Only copied 2 of 3 BC layers. Action head was random.
    Bug: Termination threshold was wrong (fired at 9m not 3m).
    Result: BC initialization was mostly useless.

v2 (ppo_bc_init_v2.zip):
    Fixed both bugs. Added shaped reward (RacingReward).
    Result: Reached +668 reward at 300k steps.
    Issue: Reward dip between 150k–250k steps (policy instability).

v3 (ppo_bc_stable.zip) — THIS FILE:
    Added entropy regularization (prevents policy collapse).
    Added cosine learning rate decay (prevents late-training oscillation).
    Reduced clip_range from 0.2 → 0.1 (more conservative updates).
    Goal: Eliminate the reward dip, smoother convergence.

═══════════════════════════════════════════════════════════════════════════════
HYPERPARAMETER DEEP DIVE
═══════════════════════════════════════════════════════════════════════════════

Each hyperparameter below is explained from first principles.
These are common interview questions — understand them, don't memorise them.

── learning_rate (cosine decay from 3e-4 to 1e-6) ─────────────────────────

    The learning rate controls how large each weight update is.
    We use a cosine DECAY schedule, not a fixed value.

    WHY 3e-4 as the starting value?
        This is the Adam optimizer's "default" learning rate.
        It's been empirically validated across thousands of RL experiments.
        3e-4 is the standard starting point for continuous control with PPO.

    WHY DECAY? (The problem it solves)
        In v2, we saw the reward dip around 150k–250k steps. This happened
        because late in training, the policy was near a good solution but
        the constant lr=3e-4 was still taking "large" steps. These steps
        occasionally moved the policy away from the good region, causing
        the temporary performance drop.

        With cosine decay: by 150k steps, lr has already dropped to ~1.5e-4.
        Smaller steps = policy stays near good solutions once found.

    WHY COSINE specifically?
        Cosine decay matches the natural "phases" of RL learning:
          Phase 1 (0–100k): Discovery. Agent is still learning basic behaviors.
                             Needs high lr. Cosine holds lr near initial value.
          Phase 2 (100–200k): Convergence. Agent is refining good trajectories.
                               Needs rapidly decreasing lr. Cosine drops fast here.
          Phase 3 (200–300k): Fine-tuning. Agent makes micro-improvements.
                               Needs near-zero lr. Cosine is nearly flat.
        Linear decay cuts equally throughout all phases, which is wrong.

── n_steps = 2048 ─────────────────────────────────────────────────────────

    Number of environment steps to collect before each PPO update.

    Why 2048 and not 512 or 8192?

    Too small (e.g. 512):
        - Each rollout is short → advantage estimates are high-variance
          (Monte Carlo estimates need enough timesteps to be accurate)
        - Policy updates more frequently but with noisy gradients
        - Can lead to erratic behavior

    Too large (e.g. 8192):
        - Slower to update (you wait longer before each gradient step)
        - Memory intensive
        - "Off-policy" drift: by the time you update, your first collected
          samples are "stale" (collected by a now-different policy)

    2048 is the SB3 recommended default for continuous control.
    At our env's 7000 steps/sec, 2048 steps takes ~0.3 seconds.

── batch_size = 64 ─────────────────────────────────────────────────────────

    CHANGED from 256 → 64 for stability.

    PPO collects n_steps=2048 samples, then shuffles them and takes
    minibatch gradient steps of size `batch_size`.

    Number of minibatches = n_steps / batch_size = 2048 / 64 = 32

    Why reduce batch_size?
        Smaller batches introduce MORE gradient noise (stochastic gradient).
        This seems bad, but in RL it has a regularizing effect:
        - Noisy gradients prevent the policy from over-fitting to a specific
          batch of trajectories from the current rollout
        - Helps escape shallow local optima
        - Standard recommendation for PPO stability

        With batch_size=256, we had 8 minibatches — the policy updated based
        on large, smooth gradient estimates that could make big committed steps.
        With batch_size=64, we have 32 minibatches — more updates per rollout,
        each with noisier but more regularized gradients.

── n_epochs = 10 ───────────────────────────────────────────────────────────

    Number of passes through the 2048-step rollout buffer before discarding it.

    PPO is "on-policy" — data collected by policy π is only valid for updating π.
    After you update, π changed slightly, making old data slightly "stale."
    PPO's clip_range limits how much π can change per epoch, making
    multiple passes safe — but only up to a point.

    n_epochs=10: standard. Higher values squeeze more signal from each rollout
    but risk over-fitting to that specific batch of experiences.

── gamma = 0.99 ────────────────────────────────────────────────────────────

    The discount factor. Controls how much the agent values FUTURE rewards
    vs IMMEDIATE rewards.

    MATH:
        The agent's objective is to maximize the "return" G_t:
        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^{∞} γ^k · r_{t+k}

        With γ=0.99, a reward 100 steps in the future is worth:
        0.99^100 = 0.366 (36.6% of face value)

        A reward 500 steps in the future is worth:
        0.99^500 = 0.0066 (0.66% of face value)

    WHY 0.99 for racing?
        Our episodes can be up to 2000 steps (at dt=0.1s = 200 seconds of driving).
        We want the agent to care about rewards 10 seconds from now (100 steps):
        0.99^100 = 0.37 — yes, these future rewards still matter.

        If γ=0.9: rewards 23 steps away are worth < 10%. Too short-sighted —
        the agent would only optimize the next 2–3 seconds.

        If γ=1.0: infinite horizon, agent treats all future rewards equally.
        This causes numerical instability (returns can be very large).

── gae_lambda = 0.95 ───────────────────────────────────────────────────────

    GAE = Generalized Advantage Estimation (Schulman et al., 2016).

    The "advantage" A(s,a) tells us: "how much BETTER was this action
    compared to what we expected?" It's used to compute the policy gradient.

    Computing advantage from raw returns has HIGH VARIANCE:
        A_t = G_t - V(s_t)
        G_t involves summing many future rewards → noisy signal

    Instead, GAE computes a smoothed advantage using the value function:
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)      ← one-step TD error
        A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}  ← exponentially weighted sum

    The parameter λ controls the bias-variance tradeoff:
        λ=1.0 → Monte Carlo (unbiased, high variance) — like raw returns
        λ=0.0 → TD(0) (low variance, high bias) — uses V(s) heavily
        λ=0.95 → sweet spot for most control tasks

    Why does variance matter?
        High variance gradients → noisy policy updates → slow learning
        Too much bias → value function errors corrupt the gradient direction
        λ=0.95 gives stable, reasonably unbiased advantage estimates.

── clip_range = 0.1 ────────────────────────────────────────────────────────

    CHANGED from 0.2 → 0.1. This is the core PPO mechanism.

    PPO's clipped objective:
        r_t(θ) = π_θ(a|s) / π_θ_old(a|s)     ← probability ratio

        L_CLIP = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]

        where ε = clip_range.

    What this means in plain English:
        When updating the policy, we compute HOW MUCH more/less likely the
        new policy is to take the same action vs the old policy. If this ratio
        would become too extreme (> 1+ε or < 1-ε), we CLIP it — we refuse
        to make that large an update.

        ε=0.2 means: allow up to 20% more/less likely per update step.
        ε=0.1 means: allow up to 10% more/less likely per update step.

    WHY REDUCE FROM 0.2 → 0.1?
        We're using BC initialization. The policy STARTS in a good region.
        With ε=0.2, each PPO update can drift 20% from the BC prior.
        Over 10 epochs of updates per rollout, this can compound into
        large policy changes that destroy the BC-learned behavior.

        ε=0.1 forces more conservative updates. The policy improves gradually,
        preserving the good BC prior while refining it with RL.

        Trade-off: slower early learning (less aggressive updates)
        vs more stability (less likely to destroy good behavior).
        For BC-initialized policies, stability wins.

── ent_coef = 0.005 ────────────────────────────────────────────────────────

    CHANGED from 0.0 → 0.005. This adds entropy regularization.

    WHAT IS ENTROPY?
        In information theory, entropy H measures the "randomness" or
        "spread" of a probability distribution.

        For a Gaussian policy with standard deviation σ:
        H(π) ≈ 0.5 · log(2πe · σ²)

        High entropy = large σ = wide distribution = EXPLORATORY policy
        Low entropy = small σ = narrow distribution = DETERMINISTIC policy

    THE PROBLEM WITH ZERO ENTROPY BONUS (ent_coef=0.0):
        PPO naturally collapses the policy distribution over time:
        - It finds actions that give good reward
        - It increases the probability of those actions
        - σ shrinks → entropy decreases → policy becomes overconfident
        - Once overconfident, a small gradient perturbation causes a LARGE
          relative change in action probabilities → instability

        This is what caused the reward dip at ~150k steps in v2.
        The policy got too narrow (low entropy), then one update pushed
        it slightly in the wrong direction, and the narrow distribution
        couldn't recover quickly.

    THE FIX (ent_coef=0.005):
        PPO's loss function becomes:
        L = L_CLIP + vf_coef · L_VF - ent_coef · H(π)

        The "-ent_coef · H(π)" term PENALIZES LOW ENTROPY.
        It adds a gradient that pushes σ to stay larger.
        This keeps the policy "exploratory enough" to avoid getting stuck.

    WHY 0.005 specifically?
        This value is a small but meaningful signal:
        - Large enough to prevent entropy collapse (σ from going near-zero)
        - Small enough not to dominate the reward signal
        - 0.005 is 0.5% of the reward scale — it's a gentle regularizer
        Typical values in literature: 0.001 – 0.01 for continuous control.

        Too large (e.g. 0.1): the agent maximizes entropy (randomness) instead
        of reward. Policy becomes random again, destroying BC initialization.
        Too small (e.g. 0.0001): insufficient effect, entropy still collapses.

── vf_coef = 0.5 ───────────────────────────────────────────────────────────

    ADDED explicitly (was using SB3 default of 0.5).

    The full PPO loss has three components:
        L = L_CLIP - ent_coef · H(π) + vf_coef · L_VF

        L_CLIP:  policy gradient loss (makes good actions more likely)
        H(π):    entropy (keeps policy exploratory)
        L_VF:    value function loss = MSE(V(s), actual_return)
                 (makes the critic better at predicting returns)

    vf_coef controls how much the VALUE FUNCTION LOSS is weighted
    relative to the policy gradient loss.

    0.5 means: "the critic's learning is half as important as the actor's."

    WHY NOT vf_coef=1.0?
        The value function is only a tool — it's used to compute advantages.
        If we make it too important, we spend more gradient budget improving
        the critic at the expense of the actor. 0.5 is the standard balance.

    WHY NOT vf_coef=0.0?
        The critic helps reduce gradient variance via GAE.
        No critic = pure Monte Carlo policy gradient = very noisy updates.

── max_grad_norm = 0.5 ─────────────────────────────────────────────────────

    ADDED explicitly (SB3 default is 0.5).

    Gradient clipping: if the gradient vector's L2 norm exceeds max_grad_norm,
    it is rescaled so its norm = max_grad_norm.

    MATH:
        If ||g|| > max_grad_norm:
            g = g * (max_grad_norm / ||g||)

    WHY?
        In RL, occasional "bad" rollouts (the agent crashes spectacularly)
        can produce very large gradients that push the policy far in the
        wrong direction in one step — "exploding gradients."

        Clipping to norm=0.5 prevents any single gradient update from being
        catastrophically large, regardless of the rollout content.

        Think of it as a speed limit on how fast the policy can change
        per gradient step.
"""

import sys
import math
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.bc_init_policy import load_bc_weights_into_ppo, verify_transfer
from rl.make_env import make_env
from rl.schedules import cosine_schedule


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK — Log extra metrics to TensorBoard at each rollout
# ─────────────────────────────────────────────────────────────────────────────

class RacingMetricsCallback(BaseCallback):
    """
    WHAT IS A CALLBACK?
    ===================
    In SB3, a "callback" is a piece of code that SB3 automatically calls
    at specific points during training. Think of it as a hook — you don't
    call it yourself, SB3 does.

    The key callback events:
        on_rollout_end():  called after collecting n_steps of experience
        on_step():         called after each individual environment step
        on_training_end(): called when learn() finishes

    WHY DO WE NEED A CUSTOM CALLBACK HERE?
    =======================================
    SB3 automatically logs reward and episode length to TensorBoard.
    But it doesn't log our domain-specific metrics:
        - Mean speed (m/s)
        - Mean lateral error (meters)
        - Entropy of the policy distribution

    To track these and compare policies properly, we need to extract them
    from the rollout buffer ourselves.

    HOW DOES THIS WORK TECHNICALLY?
    ================================
    After each rollout, `self.model.rollout_buffer` contains the collected
    transitions. The `infos` from each step are stored in the rollout.

    SB3 stores episode infos in `self.model.ep_info_buffer` — a deque of
    the most recent completed episodes' info dicts.

    We can access these to compute mean metrics across recent episodes.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Running lists to track metrics since last log
        self._ep_speeds = []        # mean speed per episode (m/s)
        self._ep_lat_errors = []    # mean |lateral error| per episode (m)

    def _on_step(self) -> bool:
        """
        Called at every environment step.

        self.locals["infos"] is a list of info dicts from the vectorized env.
        Each info dict has our custom keys: "speed", "lateral_error", "heading_error".

        When an episode ends, SB3 adds an "episode" key to the info dict
        with {"r": total_reward, "l": episode_length, "t": time_elapsed}.
        We use this as a signal that an episode just completed.

        Returns True to continue training (returning False would stop training).
        """
        for info in self.locals["infos"]:
            # Accumulate per-step speed and lateral error
            if "speed" in info:
                self._ep_speeds.append(info["speed"])
            if "lateral_error" in info:
                self._ep_lat_errors.append(abs(info["lateral_error"]))

        return True   # always continue training

    def _on_rollout_end(self) -> None:
        """
        Called after collecting n_steps of experience (i.e. after each full rollout).

        This is where we log our custom metrics to TensorBoard.
        self.logger is SB3's TensorBoard logger — we can write to it with
        self.logger.record("key", value).
        """
        if self._ep_speeds:
            # Average speed across all steps in this rollout
            mean_speed = sum(self._ep_speeds) / len(self._ep_speeds)
            # un-normalize: speed was stored as raw m/s from info dict (already un-normalized)
            self.logger.record("racing/mean_speed_ms", mean_speed)

        if self._ep_lat_errors:
            mean_lat = sum(self._ep_lat_errors) / len(self._ep_lat_errors)
            self.logger.record("racing/mean_lateral_error_m", mean_lat)

        # Log current learning rate — useful to verify cosine decay is working
        # The optimizer has param_groups, each with a 'lr' key
        current_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.logger.record("train/current_lr", current_lr)

        # Log policy entropy: high = exploratory, low = overconfident
        # SB3 stores the last computed entropy in model.logger
        # We can compute it from the distribution if available
        # (SB3 also logs entropy_loss automatically, but we log the raw value too)

        # Reset buffers for next rollout
        self._ep_speeds.clear()
        self._ep_lat_errors.clear()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train():
    """
    Full PPO training pipeline: BC init + stability improvements.

    The three stability improvements vs v2:
        1. ent_coef=0.005     — prevents policy entropy collapse
        2. cosine lr schedule — prevents late-training oscillation
        3. clip_range=0.1     — more conservative updates for warm-started policy
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # ── Total training budget ────────────────────────────────────────────────
    # 300k steps is our standard. At ~7000 steps/sec, this takes ~45 seconds.
    # In production robotics, you'd train for 10M–100M steps.
    TOTAL_TIMESTEPS = 300_000

    # ── Build vectorized environment ─────────────────────────────────────────
    # DummyVecEnv wraps a single F1Env in SB3's expected vectorized interface.
    # The [make_env] list means "1 parallel environment."
    # For 8x faster training: DummyVecEnv([make_env] * 8) — runs 8 envs in parallel.
    # We use 1 here to keep results directly comparable with our previous runs.
    env = DummyVecEnv([make_env])

    # ── Build PPO model ──────────────────────────────────────────────────────
    # Every parameter here is explained in the module docstring above.
    model = PPO(
        policy="MlpPolicy",     # Standard MLP actor-critic. SB3 handles distribution math.
        env=env,

        # ── STABILITY IMPROVEMENT 1: Cosine LR decay ────────────────────────
        # Instead of a fixed float, we pass a FUNCTION that SB3 calls each update.
        # cosine_schedule(3e-4) returns a closure: f(progress_remaining) → lr
        # See rl/schedules.py for the full derivation.
        learning_rate=cosine_schedule(initial_lr=3e-4, min_lr=1e-6),

        # Rollout collection and minibatch settings
        n_steps=2048,           # steps collected before each PPO update
        batch_size=64,          # CHANGED: 256→64 for more gradient noise / regularization
        n_epochs=10,            # passes over each rollout buffer

        # Return and advantage estimation
        gamma=0.99,             # discount factor (values future rewards)
        gae_lambda=0.95,        # GAE lambda (bias-variance tradeoff for advantages)

        # ── STABILITY IMPROVEMENT 2: Tighter clip range ──────────────────────
        # CHANGED from 0.2 → 0.1.
        # With BC warm start, policy starts in a GOOD region.
        # Smaller clip = we preserve BC prior while fine-tuning with RL.
        clip_range=0.1,

        # ── STABILITY IMPROVEMENT 3: Entropy regularization ─────────────────
        # CHANGED from 0.0 → 0.005.
        # Adds -0.005 * H(π) to the loss, penalizing low entropy.
        # Prevents the policy distribution from collapsing to near-deterministic,
        # which caused the reward dip in v2 between 150k–250k steps.
        ent_coef=0.005,

        # Value function and gradient settings
        vf_coef=0.5,            # weight of critic loss relative to actor loss
        max_grad_norm=0.5,      # gradient norm clipping prevents exploding gradients

        # Architecture: same as BCPolicy (required for weight transfer)
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),

        verbose=1,
        device=device,
    )

    # ── Configure TensorBoard logging ────────────────────────────────────────
    # Writes event files to runs/ppo_bc_stable/.
    # Run: tensorboard --logdir runs/ to compare ALL runs side by side.
    model.set_logger(configure("runs/ppo_bc_stable", ["stdout", "tensorboard"]))

    # ── Transfer BC weights into PPO actor ───────────────────────────────────
    # This is the "warm start." After this call, the actor is the trained BC policy.
    # The critic starts from random weights — it will learn quickly because
    # the actor already produces reasonable trajectories with real reward signal.
    bc_path = str(project_root / "bc" / "bc_policy_final.pt")
    load_bc_weights_into_ppo(model, bc_path, device)
    verify_transfer(model, bc_path, device)

    # ── Build callback ───────────────────────────────────────────────────────
    # RacingMetricsCallback logs speed and lateral error to TensorBoard.
    # These domain-specific metrics tell us MORE than just reward:
    #   - Is the agent actually going fast? (speed)
    #   - Is it staying on the racing line? (lateral error)
    callback = RacingMetricsCallback(verbose=0)

    # ── Run PPO training ─────────────────────────────────────────────────────
    print(f"\n[Train] Starting PPO (stable) — {TOTAL_TIMESTEPS:,} steps")
    print(f"        Stability improvements: ent_coef=0.005, clip_range=0.1, cosine lr")
    print(f"        TensorBoard: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,       # our custom metrics logger
        reset_num_timesteps=True # start timestep counter from 0 for this run
    )

    # ── Save trained model ───────────────────────────────────────────────────
    save_path = str(project_root / "rl" / "ppo_bc_stable.zip")
    model.save(save_path)
    print(f"\n[Train] Saved PPO (stable) to {save_path}")


if __name__ == "__main__":
    train()
