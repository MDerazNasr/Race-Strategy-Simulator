"""
Head-to-Head Policy Evaluation: Expert vs BC vs PPO-Scratch vs PPO-Stable.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS EVALUATION IN RL? (START HERE)
═══════════════════════════════════════════════════════════════════════════════

During training, we watch the reward curve to know if learning is happening.
But reward alone doesn't tell us everything. We need to ASK:

  "How does this policy ACTUALLY BEHAVE in the environment?"

Evaluation = running a trained policy with no gradient updates, collecting
metrics about what it does, and comparing policies against each other.

The key distinction:
  TRAINING MODE:  neural network computes gradients, policy updates every rollout
  EVAL MODE:      no gradients, deterministic actions (pick the mean, no sampling),
                  just measure what the policy does

In PyTorch: policy.eval() switches off dropout, BatchNorm updates, etc.
In SB3:     model.predict(obs, deterministic=True) uses the mean action,
            not a sample from the Gaussian distribution.

WHY DETERMINISTIC=TRUE DURING EVALUATION?
==========================================
During training, PPO samples actions from a Gaussian distribution:
  action ~ N(mu(s), sigma^2)
This exploration is essential for discovering better behaviors.

During evaluation, we want to see the BEST behavior the policy has learned,
not a random sample of it. So we use the mean directly: action = mu(s).

Think of it like: during practice, an F1 driver experiments with different
lines and braking points. During the race, they commit to the best one.

WHAT METRICS ARE WE MEASURING?
================================
  lap_completion_rate:  fraction of episodes where car survived max_steps
                        without going off-track (terminated=False, truncated=True)
                        This is the cleanest measure of whether the policy is competent.

  avg_reward:           mean total reward per episode. Combines speed + stability.

  avg_speed_ms:         mean speed in meters/second (real units, not normalized).
                        Higher is better — F1 is about going fast.

  avg_lateral_error_m:  mean |lateral error| from centerline in METERS.
                        Lower is better — measures track-following precision.

  avg_steps:            mean episode length. Combined with lap_completion_rate,
                        this tells you HOW LONG the car stays alive.

  laps_completed:       mean number of full laps completed per episode.
                        A "lap" = track index wrapping from end back to start.

WHY DO WE RUN MULTIPLE EPISODES (N=20)?
========================================
A single episode is not representative — random initial conditions mean
one episode might start in a favorable position and another in an unfavorable one.

By averaging over N=20 episodes, we get a STABLE estimate of each policy's
true performance. This is the same reason we report mean ± std in papers.

In production robotics, you'd run 100-1000 episodes per evaluation.

═══════════════════════════════════════════════════════════════════════════════
THE FOUR POLICIES BEING COMPARED
═══════════════════════════════════════════════════════════════════════════════

1. EXPERT (rule-based)
   How it works: Hand-coded controller. Looks N waypoints ahead, computes
   target angle, applies proportional-derivative steering.
   Strengths: Deterministic, always stable, good baseline.
   Weaknesses: No learning, can't improve, limited to what the programmer anticipated.
   Real-world equivalent: A pre-programmed industrial robot arm.

2. BEHAVIORAL CLONING (BC)
   How it works: Supervised learning from expert demonstrations.
   Input: observation → Output: action that mimics the expert.
   Strengths: Easy to train, good starting point.
   Weaknesses: Distribution shift — if the agent drifts from expert states,
   it has never seen those states in training and can behave erratically.
   Real-world equivalent: A self-driving car trained only on recorded
   human drives, with no online experience.

3. PPO FROM SCRATCH (v2)
   How it works: Pure reinforcement learning, random initialization.
   No demonstrations, learns only from environment reward signal.
   Strengths: Can discover novel strategies the expert never showed.
   Weaknesses: Cold start problem — random policy almost always crashes,
   giving almost no useful gradient signal early on. Got stuck at -113 reward.
   Real-world equivalent: A baby learning to walk with no guidance.

4. PPO + BC INIT + STABILITY FIXES (stable)
   How it works: BC warm start + entropy regularization + cosine LR + tight clip.
   Strengths: Best of both worlds — starts competent (BC), then improves (RL).
   Expected to outperform all others on all metrics.
   Real-world equivalent: Training a student driver using both a human instructor
   (BC demonstrations) AND on-road practice with feedback (RL rewards).
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env
from expert.expert_driver import ExpertDriver
from bc.train_bc import BCPolicy
from stable_baselines3 import PPO


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """
    Stores metrics from a SINGLE rollout episode.

    We use a @dataclass here — Python automatically generates __init__,
    __repr__, and other boilerplate. It's cleaner than a plain dict because:
      - Type hints make it self-documenting
      - Attribute access (result.total_reward) is cleaner than dict access
      - Can add methods if needed later

    All speed and lateral error values are in REAL units (m/s and meters),
    NOT the normalized values from the observation vector.
    """
    steps: int              # how many steps the episode lasted
    total_reward: float     # sum of all rewards received
    terminated: bool        # True = went off-track, False = survived max_steps
    mean_speed_ms: float    # mean speed in m/s
    mean_lateral_error_m: float   # mean |lateral error| in meters
    max_lateral_error_m: float    # worst single-step lateral error in meters
    laps_completed: int     # number of full laps completed
    trajectory_x: List[float] = field(default_factory=list)  # for trajectory plot
    trajectory_y: List[float] = field(default_factory=list)


@dataclass
class PolicySummary:
    """
    Aggregated statistics across N episodes for ONE policy.
    """
    name: str
    color: str                      # for consistent plot coloring
    lap_completion_rate: float      # fraction of episodes that didn't crash
    avg_reward: float
    std_reward: float               # standard deviation across episodes
    avg_speed_ms: float
    avg_lateral_error_m: float
    avg_steps: float
    avg_laps_completed: float
    all_results: List[EpisodeResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: F1Env,
    policy_fn: Callable,
    record_trajectory: bool = False,
    fixed_start: bool = False,
) -> EpisodeResult:
    """
    Run ONE episode with a given policy and collect metrics.

    Args:
        env:                 A raw F1Env instance (NOT wrapped in DummyVecEnv).
                             We use the raw env so we can access env.car for the expert.

        policy_fn:           A callable that takes (obs, env) and returns an action array.
                             Signature: policy_fn(obs: np.ndarray, env: F1Env) -> np.ndarray

                             Why pass env? Because the Expert policy needs env.car to
                             read x, y, yaw, v directly. BC and PPO policies only need obs.
                             Using (obs, env) gives a unified interface for all policy types.

        record_trajectory:   If True, store (x, y) position at each step.
                             Used to generate the trajectory comparison plot.
                             Turned OFF for bulk evaluation (saves memory).

    Returns:
        EpisodeResult with all metrics for this single episode.

    HOW LAP COUNTING WORKS:
        Previously this function recomputed the track index every step using
        closest_point() and detected wrap-around locally.  Now F1Env.step()
        does this computation itself and stores the result in info["laps_completed"]
        and env.laps_completed.  We read env.laps_completed at the end of the
        episode — one source of truth, no duplicated logic.

    FIXED VS RANDOM START:
        fixed_start=False (default): env.reset() picks a random track position,
            random heading (±10°), and random speed (2–6 m/s).  This tests
            robustness across all starting conditions.

        fixed_start=True: env.reset() places the car at waypoint 0, aligned
            with the track, at 5 m/s.  All policies start identically — this
            is the fair comparison that removes random-start bias against fast
            policies.
    """
    reset_opts = {"fixed_start": True} if fixed_start else None
    obs, info = env.reset(options=reset_opts)

    # Accumulate metrics across all steps in this episode
    total_reward = 0.0    # running sum of rewards — this is the key metric
    speeds = []
    lateral_errors = []
    trajectory_x = []
    trajectory_y = []

    terminated = False
    truncated  = False
    step = 0

    for step in range(env.max_steps):
        # ── Get action from the policy ─────────────────────────────────────
        # All policies receive (obs, env). Most ignore env; expert uses env.car.
        action = policy_fn(obs, env)

        # ── Step the environment ───────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)

        # ── Accumulate reward ──────────────────────────────────────────────
        # THIS is the running sum — total_reward is the sum of ALL step rewards.
        # Each step's reward is ~[-1, +1]. Over 2000 steps at full speed that
        # can reach +2000 (plus up to N × 100 lap bonuses).
        # Over 100 steps crashing that's around -20 to -100.
        # This is the standard RL "return" (undiscounted for evaluation purposes).
        total_reward += reward

        # ── Collect per-step metrics ───────────────────────────────────────
        # info["speed"] and info["lateral_error"] are in REAL UNITS (m/s, meters)
        # because we updated f1_env.py to un-normalize them in the info dict.
        speeds.append(info["speed"])
        lateral_errors.append(abs(info["lateral_error"]))

        # ── Trajectory recording (optional) ───────────────────────────────
        if record_trajectory:
            trajectory_x.append(env.car.x)
            trajectory_y.append(env.car.y)

        # ── Check episode end ──────────────────────────────────────────────
        if terminated or truncated:
            break

    return EpisodeResult(
        steps=step + 1,
        total_reward=total_reward,
        terminated=terminated,
        mean_speed_ms=float(np.mean(speeds)) if speeds else 0.0,
        mean_lateral_error_m=float(np.mean(lateral_errors)) if lateral_errors else 0.0,
        max_lateral_error_m=float(np.max(lateral_errors)) if lateral_errors else 0.0,
        laps_completed=env.laps_completed,   # read from env — single source of truth
        trajectory_x=trajectory_x,
        trajectory_y=trajectory_y,
    )


def run_episodes(
    env: F1Env,
    policy_fn: Callable,
    policy_name: str,
    policy_color: str,
    n_episodes: int = 20,
    record_one_trajectory: bool = True,
    fixed_start: bool = False,
) -> PolicySummary:
    """
    Run N episodes and aggregate results into a PolicySummary.

    WHY N=20 EPISODES?
        One episode tells you almost nothing — results vary hugely based on
        random initial position and speed. 20 episodes gives you a stable mean.
        (In papers, you'd use 50-100 for publishable results.)

    Args:
        env:                    Raw F1Env
        policy_fn:              Callable (obs, env) -> action
        policy_name:            String label for tables and plots
        policy_color:           Hex color string for plots
        n_episodes:             How many episodes to average over
        record_one_trajectory:  Record the trajectory of the LAST episode for plotting

    Returns:
        PolicySummary with mean/std of all metrics
    """
    results = []

    for ep in range(n_episodes):
        # Record trajectory on the last episode only (saves memory for N-1 episodes)
        record = record_one_trajectory and (ep == n_episodes - 1)
        result = run_episode(env, policy_fn, record_trajectory=record, fixed_start=fixed_start)
        results.append(result)

    # ── Aggregate across episodes ──────────────────────────────────────────
    rewards        = [r.total_reward          for r in results]
    speeds         = [r.mean_speed_ms         for r in results]
    lat_errors     = [r.mean_lateral_error_m  for r in results]
    steps_list     = [r.steps                 for r in results]
    laps           = [r.laps_completed        for r in results]
    completions    = [not r.terminated        for r in results]  # True = survived

    return PolicySummary(
        name=policy_name,
        color=policy_color,
        lap_completion_rate=float(np.mean(completions)),   # fraction in [0, 1]
        avg_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        avg_speed_ms=float(np.mean(speeds)),
        avg_lateral_error_m=float(np.mean(lat_errors)),
        avg_steps=float(np.mean(steps_list)),
        avg_laps_completed=float(np.mean(laps)),
        all_results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POLICY FACTORY FUNCTIONS
# Each returns a callable: policy_fn(obs, env) -> action
# ─────────────────────────────────────────────────────────────────────────────

def make_expert_policy(env: F1Env) -> Callable:
    """
    Build a policy function that uses the hand-coded ExpertDriver.

    The ExpertDriver is a RULE-BASED CONTROLLER, not a neural network.
    It uses geometry: finds the track waypoint N steps ahead, computes
    the angle to it, and applies proportional steering.

    Why does it need env directly?
        ExpertDriver.get_action(car) reads car.x, car.y, car.yaw, car.v
        from the Car object. This information IS IN the observation vector
        but in normalized form. The expert was written to use raw car state.

    We could rewrite the expert to use the obs vector — that's a refactor
    for Week 3. For now, we access env.car directly.
    """
    expert = ExpertDriver(
        track=env.track,
        lookahead=8,         # look 8 waypoints ahead when steering
        max_speed=20.0,      # target speed m/s
        corner_factor=12.0,  # how much to slow in corners
    )

    def policy_fn(obs: np.ndarray, env: F1Env) -> np.ndarray:
        # Expert ignores obs — uses car state directly
        # This is fine for evaluation (both contain the same information)
        return expert.get_action(env.car)

    return policy_fn


def make_bc_policy(model_path: str, device: str) -> Callable:
    """
    Load a trained BC neural network and wrap it as a policy function.

    The BC policy is a PyTorch MLP: 6-dimensional obs → 2-dimensional action.
    We need to:
      1. Load the saved weights from disk
      2. Set the model to eval() mode (turns off dropout, gradient tracking)
      3. Wrap it in a function that handles numpy↔torch conversion

    NUMPY VS TORCH — WHY THE CONVERSION?
        The environment returns observations as numpy arrays (CPU arrays).
        PyTorch neural networks expect torch.Tensor objects.
        After inference, we convert back to numpy for the environment.

        obs (numpy) → torch.FloatTensor → BCPolicy forward pass → numpy action

    TORCH.NO_GRAD() — WHY?
        During inference, we don't need PyTorch to track gradients.
        torch.no_grad() is a context manager that disables gradient tracking.
        This is ~30% faster and uses less memory than running with gradients.
        ALWAYS use it during evaluation — forgetting this is a common bug.

    UNSQUEEZE(0) / SQUEEZE(0) — WHY?
        The BC network expects batched input: shape (batch_size, 6).
        For a single observation: shape is (6,).
        unsqueeze(0) adds a batch dimension: (6,) → (1, 6).
        After inference, squeeze(0) removes it: (1, 2) → (2,).
    """
    # Auto-detect state_dim from saved weights (same pattern as bc_init_policy.py).
    # 'net.0.weight' has shape (hidden, state_dim) — axis 1 is input size.
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dim  = ckpt["net.0.weight"].shape[1]   # 11 after Part B+A
    action_dim = ckpt["net.4.weight"].shape[0]   # always 2
    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim)
    policy.load_state_dict(ckpt)
    policy.eval()     # IMPORTANT: eval mode disables training-specific behavior
    policy.to(device)

    def policy_fn(obs: np.ndarray, env: F1Env) -> np.ndarray:
        # env is ignored — BC only needs the observation vector
        with torch.no_grad():   # no gradient tracking during inference
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (6,) → (1,6)
            action_t = policy(obs_t)                                  # (1,2)
            action = action_t.squeeze(0).cpu().numpy()                # (1,2) → (2,)
        return action

    return policy_fn


def make_ppo_policy(model_path: str, device: str, obs_dim: int = 11) -> Callable:
    """
    Load a trained SB3 PPO model and wrap it as a policy function.

    SB3's PPO.load() reconstructs the full model from the .zip file,
    including architecture, weights, and optimizer state.

    KEY PARAMETER: deterministic=True in model.predict()
        During TRAINING: PPO samples from the Gaussian distribution.
            action ~ N(mu(s), sigma^2)  — exploration
        During EVALUATION: We use the mean directly.
            action = mu(s)              — best known behavior

        deterministic=True ensures we see the policy's BEST behavior,
        not a noisy sample. This is standard practice for evaluation.

    WHY model.predict() INSTEAD OF model.policy(obs)?
        model.predict() handles all the normalization SB3 may have applied,
        handles vectorized vs non-vectorized environments, and returns
        a numpy array. model.policy(obs) returns raw tensor and requires
        manual handling. Always use model.predict() for inference.

    Args:
        model_path:  Path to the .zip model file.
        device:      PyTorch device ('cpu' or 'cuda').
        obs_dim:     Expected observation dimension.  Default 11 (all standard
                     policies).  Use 12 for tyre-degradation policies, which
                     were trained with the 12D tyre env (obs includes tyre_life).
                     When obs_dim=12, evaluation must use a tyre env so that
                     the obs vector matches what the policy expects.
    """
    model = PPO.load(model_path, device=device)

    def policy_fn(obs: np.ndarray, env: F1Env) -> np.ndarray:
        # env is ignored — PPO only needs the observation
        # deterministic=True: use mean action (best behavior, no sampling noise)
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy_fn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(n_episodes: int = 20, fixed_start: bool = False) -> List[PolicySummary]:
    """
    Load all policies and evaluate them head-to-head.

    Args:
        n_episodes:   Number of episodes to average over per policy.
        fixed_start:  If True, all episodes start from the same position
                      (waypoint 0, track-aligned, 5 m/s).  If False (default),
                      each episode uses a random start position and heading.

    Returns a list of PolicySummary objects, one per policy.

    WHY TWO EVALUATION MODES?
        Random start (fixed_start=False):
            Tests robustness — the policy must handle ANY starting condition.
            This is the "hard" benchmark.  Fast policies (15 m/s) are
            penalised: a large heading error at high speed → crash in < 1 s.
            The slow expert (8 m/s) can recover from the same error → it scores
            higher on lap-completion rate even though it drives slower.

        Fixed start (fixed_start=True):
            All policies begin at the same waypoint with zero perturbation.
            This gives the cleanest comparison of pure driving ability.
            In real racing, starts are never random — this is the fair benchmark.
    """
    device = "cpu"   # evaluation is fast enough on CPU
    # Standard 11D env shared across all non-tyre policies.
    env = F1Env()
    # Tyre degradation env (12D obs, 2D actions) — for ppo_tyre policy.
    env_tyre = F1Env(tyre_degradation=True)
    # Pit-stop env (12D obs, 3D actions) — for ppo_pit policy (d18).
    # Separate env needed because action space must match the policy.
    env_pit = F1Env(tyre_degradation=True, pit_stops=True)

    start_label = "FIXED START" if fixed_start else "RANDOM START"
    print("=" * 60)
    print(f"POLICY EVALUATION — {start_label}")
    print(f"  {n_episodes} episodes per policy, deterministic rollouts")
    print("=" * 60)

    # ── Policy configurations ──────────────────────────────────────────────
    # Each entry: (name, color, how to build the policy_fn)
    # Colors are consistent across all plots
    configs = [
        {
            "name":   "Expert (rule-based)",
            "color":  "#44BB44",
            "fn":     lambda: make_expert_policy(env),
        },
        {
            "name":   "BC (imitation)",
            "color":  "#FFB800",
            "fn":     lambda: make_bc_policy(
                str(project_root / "bc" / "bc_policy_final.pt"), device
            ),
        },
        # NOTE: PPO Scratch (v2) was trained on the old 6D observation space.
        # It cannot run on the 11D DynamicCar env.  Removed from comparison.
        # The cold-start result (reward=-82, speed=2.9 m/s) is documented in Notes/d11.txt.
        {
            "name":   "PPO + BC + Stable",
            "color":  "#1A9FFF",
            "fn":     lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_bc_stable.zip"), device
            ),
        },
        {
            "name":          "PPO + BC + Curriculum",
            "color":         "#00C853",
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_curriculum.zip"), device
            ),
            "optional":      True,
            "optional_file": "ppo_curriculum.zip",
        },
        {
            "name":          "PPO + Curriculum v2 (3M)",
            "color":         "#FF6D00",
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_curriculum_v2.zip"), device
            ),
            "optional":      True,
            "optional_file": "ppo_curriculum_v2.zip",
        },
        {
            "name":          "PPO Multi-Lap (3M+)",
            "color":         "#E040FB",
            # Continued from ppo_curriculum_v2.zip with multi_lap env.
            # Episodes only end on crash — no 2000-step truncation.
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_multi_lap.zip"), device
            ),
            "optional":      True,
            "optional_file": "ppo_multi_lap.zip",
        },
        {
            "name":          "PPO Tyre Degradation (5M+)",
            "color":         "#00BFA5",
            # Continued from ppo_curriculum_v2.zip with tyre degradation env.
            # Obs extended 11D -> 12D (added tyre_life).  Episodes still
            # truncate at max_steps=2000 (no multi_lap, avoids d16 mistake).
            # Tyre wear: base=0.0003/step + 0.002×slip_angles/step.
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_tyre.zip"), device,
                obs_dim=12,
            ),
            "optional":      True,
            "optional_file": "ppo_tyre.zip",
        },
        {
            "name":          "PPO Pit Stops (6M+)",
            "color":         "#FF4081",
            # Trained from scratch with 3D action space [throttle, steer, pit_signal].
            # BC warm start from expert_data_pit.npz (pit-aware demonstrations).
            # Full curriculum learning (same STAGES as ppo_curriculum).
            # Pit strategy: pay -200 to reset tyre_life, gain speed for remaining steps.
            # Evaluated in pit-stop env (12D obs, 3D action space).
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_pit.zip"), device,
                obs_dim=12,
            ),
            "optional":      True,
            "optional_file": "ppo_pit.zip",
            "env_key":       "pit",   # routes to env_pit (3D action space)
        },
        {
            "name":          "PPO Pit Strategy v2 (d19)",
            "color":         "#B388FF",
            # d19 fixes all three root causes that prevented d18 from discovering pits:
            #   Fix 1: Balanced BC dataset (generate_dataset_pit_v2 — pit-only episodes).
            #          Pit-positive fraction: 0.03% (d18) → ~5% (d19).
            #   Fix 2: gamma=0.9999 (was 0.99 in d18).
            #          Pit payoff discounted to 90% instead of 0.004%.
            #   Fix 3: Stage 0 forced pits (every 500 steps, ~100k steps).
            #          Value function bootstrapped with real pit experiences
            #          before agent must signal pits itself.
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_pit_v2.zip"), device,
                obs_dim=12,
            ),
            "optional":      True,
            "optional_file": "ppo_pit_v2.zip",
            "env_key":       "pit",   # routes to env_pit (3D action space)
        },
        {
            "name":          "PPO Pit Strategy v3 (d20)",
            "color":         "#69F0AE",
            # d20 closes the implementation gaps from d19:
            #   Fix A: Weighted BC loss (pit_class_weight=1000).
            #          Effective class ratio: 1:1 (was 1:1009 after d19 filtering).
            #   Fix B: Zero-initialize pit output row after BC weight transfer.
            #          P(pit_signal > 0) = 0.5 at start instead of ≈ 0.
            #   Fix C: forced_pit_interval=50 (fires in every episode, even ep_len=50).
            #          Gradual removal: 50 → 100 → 0 across Stages 0→1→2+.
            "fn":            lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_pit_v3.zip"), device,
                obs_dim=12,
            ),
            "optional":      True,
            "optional_file": "ppo_pit_v3.zip",
            "env_key":       "pit",   # routes to env_pit (3D action space)
        },
    ]

    summaries = []

    for cfg in configs:
        # Optional policies are skipped if their .zip file doesn't exist yet.
        # Each optional config specifies its own file via "optional_file".
        # Falls back to deriving the filename from the lambda (not always possible),
        # so we require explicit "optional_file" for any optional policy.
        if cfg.get("optional"):
            opt_file = cfg.get("optional_file", "ppo_curriculum.zip")
            model_path = project_root / "rl" / opt_file
            if not model_path.exists():
                print(f"\n  Skipping: {cfg['name']} (not yet trained — {opt_file})")
                continue

        print(f"\n  Evaluating: {cfg['name']}")
        print(f"  {'─' * 40}")

        policy_fn = cfg["fn"]()   # build the policy function

        # Route each policy to the matching environment:
        #   - ppo_pit.zip  → env_pit (12D obs, 3D action space with pit signal)
        #   - ppo_tyre.zip → env_tyre (12D obs, 2D action space, no pit)
        #   - all others   → env (11D obs, 2D action space, standard)
        if cfg.get("env_key") == "pit":
            eval_env = env_pit
        elif cfg.get("optional_file") == "ppo_tyre.zip":
            eval_env = env_tyre
        else:
            eval_env = env

        summary = run_episodes(
            env=eval_env,
            policy_fn=policy_fn,
            policy_name=cfg["name"],
            policy_color=cfg["color"],
            n_episodes=n_episodes,
            record_one_trajectory=True,
            fixed_start=fixed_start,
        )
        summaries.append(summary)

        # Print per-policy summary immediately
        print(f"  Lap completion rate:   {summary.lap_completion_rate*100:.1f}%")
        print(f"  Avg reward:            {summary.avg_reward:.2f} ± {summary.std_reward:.2f}")
        print(f"  Avg speed:             {summary.avg_speed_ms:.2f} m/s")
        print(f"  Avg lateral error:     {summary.avg_lateral_error_m:.3f} m")
        print(f"  Avg steps:             {summary.avg_steps:.0f}")
        print(f"  Avg laps completed:    {summary.avg_laps_completed:.2f}")

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING — Terminal Table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(summaries: List[PolicySummary], title: str = "FINAL EVALUATION RESULTS") -> None:
    """
    Print a formatted ASCII comparison table to the terminal.

    This is what you'd show in a paper, a presentation, or an interview.
    Each row is a policy, each column is a metric.

    We also highlight the BEST value in each column with an arrow (←).
    'Best' depends on the metric:
        lap_completion_rate → higher is better
        avg_reward          → higher is better
        avg_speed_ms        → higher is better
        avg_lateral_error_m → LOWER is better  (less drift = better)
        avg_laps_completed  → higher is better
    """
    print("\n")
    print("═" * 88)
    print(f"  {title}")
    print("═" * 88)

    # Column widths
    col_w = 22
    metric_w = 26

    header = (
        f"  {'Policy':<{col_w}}"
        f"{'Lap %':>8}"
        f"{'Reward':>10}"
        f"{'Speed m/s':>11}"
        f"{'Lat Err m':>11}"
        f"{'Laps':>7}"
    )
    print(header)
    print("  " + "─" * 66)

    # Find best value per column for highlighting
    def best_idx(values, higher_is_better=True):
        return values.index(max(values) if higher_is_better else min(values))

    lap_vals   = [s.lap_completion_rate      for s in summaries]
    rew_vals   = [s.avg_reward               for s in summaries]
    spd_vals   = [s.avg_speed_ms             for s in summaries]
    lat_vals   = [s.avg_lateral_error_m      for s in summaries]
    lap_c_vals = [s.avg_laps_completed       for s in summaries]

    best_lap   = best_idx(lap_vals,   higher_is_better=True)
    best_rew   = best_idx(rew_vals,   higher_is_better=True)
    best_spd   = best_idx(spd_vals,   higher_is_better=True)
    best_lat   = best_idx(lat_vals,   higher_is_better=False)
    best_lapc  = best_idx(lap_c_vals, higher_is_better=True)

    for i, s in enumerate(summaries):
        mark_lap  = " ←" if i == best_lap  else "  "
        mark_rew  = " ←" if i == best_rew  else "  "
        mark_spd  = " ←" if i == best_spd  else "  "
        mark_lat  = " ←" if i == best_lat  else "  "
        mark_lapc = " ←" if i == best_lapc else "  "

        row = (
            f"  {s.name:<{col_w}}"
            f"{s.lap_completion_rate*100:>6.1f}%{mark_lap}"
            f"{s.avg_reward:>8.1f}{mark_rew}"
            f"{s.avg_speed_ms:>9.2f}{mark_spd}"
            f"{s.avg_lateral_error_m:>9.3f}{mark_lat}"
            f"{s.avg_laps_completed:>5.2f}{mark_lapc}"
        )
        print(row)

    print("  " + "─" * 66)
    print("  ← = best in column")
    print("═" * 88)


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING — Matplotlib Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(summaries: List[PolicySummary], save_path: str, subtitle: str = "20 Episodes per Policy") -> None:
    """
    Generate a side-by-side bar chart comparing all 4 policies across 5 metrics.

    Bar charts are the right choice here because:
      - We're comparing DISCRETE policies (not continuous values)
      - We have a small number of categories (4 policies)
      - We want to show multiple metrics simultaneously

    WHY ERROR BARS?
        Each bar's height = mean across 20 episodes.
        Error bar = ± 1 standard deviation across episodes.
        Without error bars, two similar-looking bars might be VERY different
        in practice — one might have high variance (crashes sometimes, great other times)
        while another is consistently mediocre.
        Error bars communicate reliability, not just average performance.
    """
    metrics = [
        ("Lap Completion Rate (%)",     [s.lap_completion_rate * 100 for s in summaries],
         None,                           True,  "% of episodes survived"),
        ("Avg Episode Reward",           [s.avg_reward for s in summaries],
         [s.std_reward for s in summaries],     True,  "cumulative reward/ep"),
        ("Mean Speed (m/s)",             [s.avg_speed_ms for s in summaries],
         None,                           True,  "m/s (higher = faster)"),
        ("Mean Lateral Error (m)",       [s.avg_lateral_error_m for s in summaries],
         None,                           False, "meters from centerline"),
        ("Avg Laps Completed",           [s.avg_laps_completed for s in summaries],
         None,                           True,  "laps per episode"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.patch.set_facecolor("#0D1117")

    names   = [s.name.replace(" (", "\n(") for s in summaries]
    colors  = [s.color for s in summaries]
    x       = np.arange(len(summaries))
    bar_w   = 0.6

    for ax, (title, values, stds, higher_better, ylabel) in zip(axes, metrics):
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.tick_params(colors="#CCCCCC", labelsize=8)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--", color="#444444")

        bars = ax.bar(
            x, values,
            width=bar_w,
            color=colors,
            alpha=0.85,
            edgecolor="#30363D",
            linewidth=0.5,
        )

        # Error bars (only when std is provided)
        if stds is not None:
            ax.errorbar(
                x, values,
                yerr=stds,
                fmt="none",           # no line, just caps
                color="white",
                capsize=5,
                capthick=1.5,
                linewidth=1.5,
                alpha=0.7,
            )

        # Annotate best bar
        best_val  = max(values) if higher_better else min(values)
        best_idx_ = values.index(best_val)
        ax.bar(
            [x[best_idx_]], [values[best_idx_]],
            width=bar_w,
            color=colors[best_idx_],
            edgecolor="white",
            linewidth=2.0,
        )

        # Value labels on top of bars
        for i, v in enumerate(values):
            label = f"{v:.1f}"
            ax.text(
                x[i], v + (max(values) - min(values)) * 0.03,
                label, ha="center", va="bottom",
                color="white", fontsize=8, fontweight="bold"
            )

        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, color="#CCCCCC")

        direction = "↑ better" if higher_better else "↓ better"
        ax.text(
            0.98, 0.97, direction,
            transform=ax.transAxes,
            ha="right", va="top",
            color="#888888", fontsize=8, style="italic"
        )

    fig.suptitle(
        f"Head-to-Head Policy Evaluation  |  F1 Racing Env  |  {subtitle}",
        fontsize=13, fontweight="bold", color="white", y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"[Eval] Saved bar chart → {save_path}")


def plot_trajectories(summaries: List[PolicySummary], env: F1Env, save_path: str) -> None:
    """
    Plot the trajectory each policy took during its last episode on the oval track.

    WHY IS THIS USEFUL?
        Numbers tell you HOW WELL a policy performs.
        Trajectories tell you HOW it performs — where it goes, how it corners.

        You can see:
          - Does the agent take a wide, smooth racing line? (PPO)
          - Does it stay near the centerline rigidly? (Expert)
          - Does it oscillate left-right? (BC with over-steering)
          - Does it cut corners? (reward-hacking behavior)

        In real F1 analytics, trajectory analysis is used to compare
        driver vs AI performance and find where time is lost.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("#0D1117")
    axes_flat = axes.flatten()

    # Draw the track on each subplot
    track = env.track

    for ax, summary in zip(axes_flat, summaries):
        ax.set_facecolor("#0D1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.tick_params(colors="#CCCCCC")

        # ── Draw track centerline ─────────────────────────────────────────
        # Close the loop by appending the first point
        tx = np.append(track[:, 0], track[0, 0])
        ty = np.append(track[:, 1], track[0, 1])
        ax.plot(tx, ty, "--", color="#444444", linewidth=1.0, label="Track center", zorder=1)

        # ── Draw approximate track width (±3m from center) ────────────────
        # We draw rough circles at radius±3m as track boundaries
        # Track is a circle: approximate with numpy offset
        center_x = np.mean(track[:, 0])
        center_y = np.mean(track[:, 1])
        theta = np.linspace(0, 2 * np.pi, 300)
        radius = np.sqrt((track[0, 0] - center_x)**2 + (track[0, 1] - center_y)**2)
        for r_offset, alpha in [(+3, 0.15), (-3, 0.15)]:
            ax.plot(
                center_x + (radius + r_offset) * np.cos(theta),
                center_y + (radius + r_offset) * np.sin(theta),
                "-", color="#445566", linewidth=1.0, alpha=alpha, zorder=1
            )

        # ── Draw policy trajectory ─────────────────────────────────────────
        last = summary.all_results[-1]   # the episode with recorded trajectory
        if last.trajectory_x:
            # Color the trajectory by progress (early=dark, late=bright)
            xs = np.array(last.trajectory_x)
            ys = np.array(last.trajectory_y)
            n = len(xs)

            # Draw as a gradient line by plotting segments
            for i in range(0, n - 1, max(1, n // 200)):
                alpha = 0.4 + 0.6 * (i / n)   # fade in over the episode
                ax.plot(
                    xs[i:i+2], ys[i:i+2],
                    "-", color=summary.color,
                    linewidth=1.8, alpha=alpha, zorder=2
                )

            # Mark start point
            ax.scatter(xs[0], ys[0], c="white", s=60, zorder=5, marker="o", label="Start")
            # Mark end point
            end_marker = "×" if last.terminated else "★"
            end_color  = "#FF4444" if last.terminated else "#44FF44"
            ax.scatter(
                xs[-1], ys[-1],
                c=end_color, s=80, zorder=5,
                marker="x" if last.terminated else "*",
                label=f"End ({'crash' if last.terminated else 'survived'})"
            )

        # ── Labels ────────────────────────────────────────────────────────
        ax.set_title(
            f"{summary.name}\n"
            f"Speed: {summary.avg_speed_ms:.1f} m/s  |  "
            f"Lat err: {summary.avg_lateral_error_m:.2f} m  |  "
            f"Laps: {summary.avg_laps_completed:.1f}",
            color="white", fontsize=9, fontweight="bold"
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=7, facecolor="#1C2128", edgecolor="#30363D", labelcolor="white")

    fig.suptitle(
        "Trajectory Comparison  |  One Episode per Policy  |  Track: Oval 50m radius",
        fontsize=12, fontweight="bold", color="white"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"[Eval] Saved trajectory plot → {save_path}")


def plot_episode_distributions(summaries: List[PolicySummary], save_path: str) -> None:
    """
    Box plots showing the DISTRIBUTION of rewards across all 20 episodes per policy.

    WHY BOX PLOTS INSTEAD OF JUST MEANS?
        A mean can hide important information:
          - Is the policy CONSISTENTLY good? (tight distribution)
          - Or SOMETIMES great and SOMETIMES terrible? (wide distribution)

        For a real F1 AI system, CONSISTENCY matters as much as peak performance.
        A policy with mean=500, std=400 is much less useful than
        one with mean=400, std=50 — even though the first looks "better" on average.

    Box plot anatomy:
        Center line = MEDIAN (50th percentile)
        Box edges   = 25th and 75th percentiles (the "interquartile range" = IQR)
        Whiskers    = 1.5 * IQR beyond the box edges
        Dots        = outliers (individual episodes far outside the whiskers)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.tick_params(colors="#CCCCCC")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", color="#444444")

    # Build data arrays and positions
    data    = [
        [r.total_reward for r in s.all_results]
        for s in summaries
    ]
    colors  = [s.color for s in summaries]
    labels  = [s.name.replace(" (", "\n(") for s in summaries]

    bp = ax.boxplot(
        data,
        patch_artist=True,    # fill boxes with color
        notch=False,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#AAAAAA", linewidth=1.2),
        capprops=dict(color="#AAAAAA", linewidth=1.5),
        flierprops=dict(marker="o", color="#AAAAAA", alpha=0.5, markersize=4),
    )

    # Color each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.2)

    ax.set_xticks(range(1, len(summaries) + 1))
    ax.set_xticklabels(labels, fontsize=9, color="#CCCCCC")
    ax.axhline(0, color="#555555", linestyle=":", linewidth=1, label="Zero reward")
    ax.set_title(
        "Episode Reward Distribution  |  20 Episodes per Policy\n"
        "(Box = 25th–75th percentile, Line = median, Dots = outliers)",
        color="white", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Total reward per episode", color="#AAAAAA")
    ax.legend(fontsize=9, facecolor="#1C2128", edgecolor="#30363D", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"[Eval] Saved distribution plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── MODE 1: Random-start evaluation (N=20) ────────────────────────────────
    # Every episode uses a random track position + random heading (±10°).
    # This is the hard benchmark: tests policy robustness across all starts.
    # Fast policies are penalised — high speed + bad heading → crash in < 1 s.
    # This is the distribution used during TRAINING, so it reveals real-world
    # robustness (or lack of it).
    print("\n" + "═" * 70)
    print("  EVALUATION MODE 1: RANDOM START  (N=20 episodes)")
    print("  Tests robustness across all starting conditions.")
    print("═" * 70)
    summaries_random = evaluate_all(n_episodes=20, fixed_start=False)
    print_comparison_table(summaries_random, title="RANDOM START — 20 Episodes per Policy")

    # ── MODE 2: Fixed-start evaluation (N=10) ────────────────────────────────
    # All episodes start from waypoint 0, track-aligned, at 5 m/s.
    # Every policy faces exactly the same starting condition.
    # This removes the random-start penalty for fast policies and gives the
    # fairest comparison of peak driving ability.
    # In real F1, starts are never random — this is the operationally relevant
    # benchmark.
    print("\n" + "═" * 70)
    print("  EVALUATION MODE 2: FIXED START  (N=10 episodes)")
    print("  All policies start from waypoint 0, track-aligned, v=5 m/s.")
    print("  Removes random-start bias against fast policies.")
    print("═" * 70)
    summaries_fixed = evaluate_all(n_episodes=10, fixed_start=True)
    print_comparison_table(summaries_fixed, title="FIXED START — 10 Episodes per Policy")

    # ── Plots ─────────────────────────────────────────────────────────────────
    env_for_plots = F1Env()

    # Random-start plots (existing set — now includes lap bonus in rewards)
    plot_bar_comparison(
        summaries_random,
        save_path=str(project_root / "plots" / "eval_bar_comparison.png"),
        subtitle="Random Start | 20 Episodes per Policy",
    )
    plot_trajectories(
        summaries_random,
        env=env_for_plots,
        save_path=str(project_root / "plots" / "eval_trajectories.png"),
    )
    plot_episode_distributions(
        summaries_random,
        save_path=str(project_root / "plots" / "eval_reward_distribution.png"),
    )

    # Fixed-start bar chart — the fair comparison
    # Saved separately so both views are preserved.
    plot_bar_comparison(
        summaries_fixed,
        save_path=str(project_root / "plots" / "eval_bar_comparison_fixed.png"),
        subtitle="Fixed Start | 10 Episodes per Policy",
    )

    print("\n[Eval] Plots saved:")
    print("  plots/eval_bar_comparison.png          (random start)")
    print("  plots/eval_bar_comparison_fixed.png    (fixed start — fair comparison)")
    print("  plots/eval_trajectories.png")
    print("  plots/eval_reward_distribution.png")
    print("[Eval] Done.")
