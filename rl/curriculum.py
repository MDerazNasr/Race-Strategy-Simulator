"""
Curriculum Learning for DynamicCar PPO Training.

WHAT IS CURRICULUM LEARNING?
==============================
Curriculum learning = "start easy, get harder over time."

The name comes from education: you don't teach calculus before algebra.
In RL, presenting hard tasks too early leads to the cold-start problem
all over again -- but now at a deeper level.

OUR SPECIFIC PROBLEM:
  DynamicCar has Pacejka tyre physics. At high speeds, lateral forces
  saturate and the car spins. A policy that tries to go 20 m/s from
  step 1 will spin constantly and learn nothing useful from the crashes.

  The reward signal is too sparse: crashing gives -20 but the agent
  does not know WHY (too fast? wrong steering? bad yaw entry?).
  With random starts, the agent tries to attribute blame across many
  possible causes -- it has no signal isolating speed as the problem.

THE CURRICULUM SOLUTION:
  Stage 1 -- Stability:  cap speed at ~8 m/s. At 8 m/s, tyre slip angles
    stay small -- the car behaves almost like the kinematic model. The agent
    learns to follow the track and observe the new obs dims (v_y, r)
    without the chaos of high-speed crashes.

  Stage 2 -- Speed:  unlock to ~15 m/s. Tyres now start working -- moderate
    slip angles generate real cornering force. The agent must learn to
    manage v_y and r rather than just observe them.

  Stage 3 -- Racing:  full speed, no cap. The agent must use everything
    it learned in Stages 1 and 2 to optimise lap time.

WHY THIS WORKS (theory):
  Each stage provides a DENSE reward signal. At 8 m/s the agent can
  complete full laps and collect large positive rewards. Those completed
  laps train the value function to understand track structure -- "being
  at waypoint 50 with v=8 m/s is worth ~X reward." The value function
  carries forward when speed increases, so the agent already "knows"
  what a good lap looks like. It only has to learn the new physics.

  Analogy: a racing driving school starts students on go-karts (low speed,
  forgiving). Once the student understands racing lines and braking points,
  they graduate to faster cars. The track knowledge transfers; only the
  car behaviour is new.

  Interview version:
    "Curriculum learning addresses the credit assignment problem at scale.
     By constraining the task initially, the reward signal stays dense
     and informative. As the agent builds a stable value function
     baseline, we expand the state-action space gradually, letting the
     agent transfer prior knowledge rather than restart from scratch."

GRADUATION CRITERION:
  The callback monitors lap_completion_rate = fraction of episodes that
  survived to max_steps (did NOT crash) over a rolling window of rollouts.
  When the rolling average exceeds a threshold, the agent advances to the
  next stage. The threshold is set lower for harder stages.
"""

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# CURRICULUM STAGE DEFINITION
# =============================================================================

@dataclass
class CurriculumStage:
    """
    One stage of the curriculum. Defines the physical speed cap AND
    the reward weights active during that stage.

    When the agent graduates a stage, the next stage's parameters are
    applied to the LIVE environment -- no restarting, no new model.
    The policy and value function weights carry forward unchanged.

    Fields:
        name:                Human-readable label (logged to TensorBoard)
        max_accel:           DynamicCar.max_accel (m/s^2) -- limits top speed.
                             At max_accel=6  m/s^2, terminal speed ~  8 m/s
                             At max_accel=11 m/s^2, terminal speed ~ 15 m/s
                             At max_accel=15 m/s^2, original DynamicCar limit
        reward_kwargs:       Keyword arguments forwarded to RacingReward().
                             Controls what behaviour is incentivised each stage.
        grad_lap_rate:       Minimum rolling lap completion rate to graduate.
        grad_window:         Number of consecutive rollouts to average over.
                             Larger window = more stable graduation criterion.
        forced_pit_interval: Steps between forced pits (d19 Stage 0 only).
                             0 = disabled (default, backward compatible).
                             500 = force a pit at step 500, 1000, 1500, ...
                             Used in STAGES_PIT_V2 Stage 0 to bootstrap the
                             value function with pit experiences before the
                             agent is required to signal pits itself.
    """
    name:                str
    max_accel:           float
    reward_kwargs:       Dict   = field(default_factory=dict)
    grad_lap_rate:       float  = 0.4
    grad_window:         int    = 5
    forced_pit_interval:  int   = 0    # d19: 0 = disabled (default)
    forced_pit_threshold: float = 0.0  # d21: 0.0 = disabled (default)


# =============================================================================
# STAGE DEFINITIONS
# =============================================================================

STAGES: List[CurriculumStage] = [

    # -- Stage 1: Stability ---------------------------------------------------
    # Goal: survive. Learn the track. Learn to read v_y and r as warning signs.
    # Physics: max_accel=6 caps speed at ~8 m/s. Tyre slip stays in linear
    # regime, behaves almost like the kinematic model. No tyre saturation.
    # Reward: lateral penalty doubled, speed bonus removed.
    #   -> agent is punished for drift but NOT rewarded for rushing.
    #   -> optimal behaviour: stay centred, point at the track, survive.
    CurriculumStage(
        name          = "Stage 1 -- Stability (<=8 m/s)",
        max_accel     = 6.0,
        reward_kwargs = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.0,   # no speed bonus -- don't reward rushing
            lateral_weight    = 1.0,   # 2x normal -- survival is the priority
            heading_weight    = 0.2,   # 2x normal -- point the right way
            smoothness_weight = 0.1,   # 2x normal -- learn smooth inputs early
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.5,   # graduate when 50% of episodes survive
        grad_window   = 5,
    ),

    # -- Stage 2: Speed -------------------------------------------------------
    # Goal: go faster without crashing. Learn that v_y is controllable.
    # Physics: max_accel=11 unlocks ~15 m/s. Tyres now work in the nonlinear
    # regime -- moderate slip gives maximum cornering force, but pushing too
    # hard saturates them. The agent must learn to feel the limit.
    # Reward: speed bonus re-enabled, lateral penalty back to normal.
    #   -> agent is now rewarded for speed, but still penalised for drift.
    CurriculumStage(
        name          = "Stage 2 -- Speed (<=15 m/s)",
        max_accel     = 11.0,
        reward_kwargs = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,   # reward going faster
            lateral_weight    = 0.5,   # back to normal lateral penalty
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.3,   # at higher speed, some crashing is acceptable
        grad_window   = 5,
    ),

    # -- Stage 3: Racing ------------------------------------------------------
    # Goal: maximise lap time. No speed cap. Same hyperparameters as the
    # stable training run from d11/d12.
    # This stage never graduates -- it runs until total_timesteps is reached.
    CurriculumStage(
        name          = "Stage 3 -- Full Racing (no cap)",
        max_accel     = 15.0,
        reward_kwargs = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 1.1,   # impossible threshold -- never graduates
        grad_window   = 9999,
    ),
]


# =============================================================================
# PIT STRATEGY V2 CURRICULUM  (d19)
# =============================================================================
#
# PROBLEM (d18 failure analysis):
#   The d18 agent never discovered pitting despite a BC warm-start.
#   Three root causes:
#     1. BC class imbalance: 3,400:1 no-pit to pit-positive samples.
#        MSE loss dominated by "no-pit" class → BC policy pushes pit_signal
#        toward -1.0 at initialisation.
#     2. gamma=0.99 discounts 1000-step future gains to near zero.
#        0.99^1000 ≈ 4e-5. The value function cannot see the tyre benefit.
#     3. Entropy collapse: by Stage 3, std(pit_signal) ≈ 0.54 centred at -1.0.
#        P(pit_signal > 0) ≈ 0. The agent never explores pitting at all.
#
# FIXES (d19):
#   Fix 1 (handled in train_ppo_pit_v2.py): balanced BC dataset.
#     generate_dataset_pit_v2() only keeps episodes where at least 1 pit fired.
#     Pit-positive fraction rises from ~0.03% to ~3-5%.
#
#   Fix 2 (handled in train_ppo_pit_v2.py): gamma=0.9999.
#     0.9999^1000 ≈ 0.905. The value function now sees 90% of rewards 1000
#     steps away — the pit payoff is visible to the gradient signal.
#
#   Fix 3 (THIS CURRICULUM): forced pit exploration Stage 0.
#     Even with gamma=0.9999, the value function needs ACTUAL pit experiences
#     to bootstrap Q(s,pit). If the agent never pits, it never learns the value.
#     Stage 0 forces a pit every 500 steps for ~100k steps.
#     After Stage 0 the agent has seen "tyre reset → more reward" many times.
#     The value function now knows Q(s,pit) > Q(s,no-pit) when tyres are worn.
#     Stage 1–3 are identical to STAGES — the agent drives freely and must
#     signal pits itself, but now it has a learned reason to do so.
#
# STAGE SCHEDULE:
#   Stage 0: ~100k steps  (50 rollouts × 2048)  forced pits every 500 steps
#   Stage 1: graduates at 50% lap rate           stability, low speed
#   Stage 2: graduates at 30% lap rate           speed unlocked
#   Stage 3: runs until total_timesteps          full racing + pit strategy
#
STAGES_PIT_V2: List[CurriculumStage] = [

    # -- Stage 0: Forced Pit Exploration -------------------------------------
    # Goal: bootstrap the value function with pit experiences.
    #
    # forced_pit_interval=500: env fires a pit at step 500 and step 1000 of
    # every episode.  The agent sees tyre_life reset to 1.0 and gains more
    # reward in the second half of the episode.  The value function learns:
    #   Q(s_500, pit) >> Q(s_500, no-pit)
    # even before the agent learns to signal pit_signal > 0.
    #
    # Graduation: grad_lap_rate=0.0 means ANY survival rate passes.
    # grad_window=50 → must see 50 rollouts before graduating.
    # 50 × 2048 ≈ 102,400 steps — enough for the value function to bootstrap.
    #
    # Speed cap: max_accel=6 (same as Stage 1).  Keep the car controllable
    # while the value function is learning the pit benefit.
    CurriculumStage(
        name                = "Stage 0 -- Forced Pit Exploration",
        max_accel           = 6.0,
        forced_pit_interval = 500,   # force pit at step 500, 1000, 1500, ...
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.0,
            lateral_weight    = 1.0,
            heading_weight    = 0.2,
            smoothness_weight = 0.1,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.0,    # graduate after filling the window
        grad_window   = 50,     # 50 rollouts × 2048 steps ≈ 100k steps
    ),

    # -- Stage 1: Stability (same as STAGES Stage 1, forced pits OFF) --------
    CurriculumStage(
        name                = "Stage 1 -- Stability (<=8 m/s)",
        max_accel           = 6.0,
        forced_pit_interval = 0,     # agent must signal pits
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.0,
            lateral_weight    = 1.0,
            heading_weight    = 0.2,
            smoothness_weight = 0.1,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.5,
        grad_window   = 5,
    ),

    # -- Stage 2: Speed (same as STAGES Stage 2) -----------------------------
    CurriculumStage(
        name                = "Stage 2 -- Speed (<=15 m/s)",
        max_accel           = 11.0,
        forced_pit_interval = 0,
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.3,
        grad_window   = 5,
    ),

    # -- Stage 3: Racing (same as STAGES Stage 3) ----------------------------
    CurriculumStage(
        name                = "Stage 3 -- Full Racing (no cap)",
        max_accel           = 15.0,
        forced_pit_interval = 0,
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 1.1,   # impossible threshold -- never graduates
        grad_window   = 9999,
    ),
]


# =============================================================================
# PIT STRATEGY V3 CURRICULUM  (d20)
# =============================================================================
#
# PROBLEM (d19 post-mortem):
#   d19's three fixes were correct in design but had two implementation gaps:
#
#   Gap 1 — Stage 0 forced pits never fired:
#     forced_pit_interval=500 requires ep_len > 500.
#     During Stage 0, ep_len ≈ 46-50.  Agent crashed before step 500.
#     The value function NEVER saw a pit experience in Stage 0.
#     Fix: use interval=50 → fires even in 50-step episodes (at step 50).
#
#   Gap 2 — BC still initialises pit_signal ≈ -1.0 (before weight transfer):
#     Even with pit_class_weight=1000 in BC training, the BC hidden layers
#     learn to associate driving-state features with "no-pit" because 99.9%
#     of the training data is no-pit.  The BC output head for pit_signal
#     still starts biased toward -1.0 after transfer.
#     Fix: zero_pit_signal_output() zeros the pit row of action_net AFTER
#     BC weight transfer, giving P(pit_signal > 0) = 0.5 at initialisation.
#
# THE GRADUAL REMOVAL SCHEDULE:
#   The key insight: forced pits help the value function but HURT the policy
#   if kept too long.  If the env always pits for the agent, the agent never
#   has to learn WHEN to send pit_signal > 0.
#
#   Stage 0 (50-step interval, ~100k steps):
#     Fires a pit every 50 steps.  Even in 50-step episodes, the very first
#     episode sees a forced pit at step 50.  Value function immediately learns:
#       Q(s_50, any_action) gets a tyre reset → more reward → Q increases.
#     The pit row (zero-initialised) starts at P(pit)=0.5 and receives gradient.
#
#   Stage 1 (100-step interval):
#     Forced pits still active but half as frequent.  The agent has to cover
#     the gaps between forced pits on its own.  Pit_signal gradient from
#     bootstrapped value function starts pulling pit_signal toward +1 at the
#     right times.
#
#   Stage 2 (no forced pits):
#     Agent fully responsible for pit decisions.  By now, the value function
#     knows Q(s, pit_signal>0) >> Q(s, pit_signal<0) when tyre_life is low.
#     The policy has been getting gradient signal from ~200k steps of forced
#     pit experiences.
#
#   Stage 3 (no forced pits, full speed):
#     Racing pace with learned pit strategy.  Agent should now pit at ~step
#     1000 when tyre_life approaches 0.3.
#
STAGES_PIT_V3: List[CurriculumStage] = [

    # -- Stage 0: Forced pit every 50 steps (fires in short early episodes) --
    # Goal: bootstrap value function with guaranteed pit experiences from step 1.
    #
    # WHY interval=50:
    #   In d19, interval=500 never fired because ep_len ≈ 46-50.
    #   interval=50 fires at step 50, 100, 150, ... ensuring that even
    #   episodes ending at step 50 see exactly one forced pit (at step 50).
    #   The value function immediately learns: "pitting at step 50 → tyre
    #   reset → grip restored → more future reward per step."
    #
    # Graduation: same as STAGES_PIT_V2 Stage 0 — after 50 rollouts.
    CurriculumStage(
        name                = "Stage 0 -- Forced Pit Bootstrap (every 50 steps)",
        max_accel           = 6.0,
        forced_pit_interval = 50,    # fires at step 50, 100, 150, ...
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.0,
            lateral_weight    = 1.0,
            heading_weight    = 0.2,
            smoothness_weight = 0.1,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.0,   # graduate after filling the window
        grad_window   = 50,    # 50 rollouts × 2048 ≈ 100k steps
    ),

    # -- Stage 1: Forced pit every 100 steps (half frequency) ----------------
    # Goal: gradually reduce training wheels while keeping pit experiences.
    # Agent must pit on its own between forced pits.  The pit_signal row,
    # initialized at 0 (P=0.5), now has ~100k steps of value-function
    # gradient pushing it toward the right decisions.
    CurriculumStage(
        name                = "Stage 1 -- Stability + Guided Pits (every 100 steps)",
        max_accel           = 6.0,
        forced_pit_interval = 100,   # fires at step 100, 200, ...
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.0,
            lateral_weight    = 1.0,
            heading_weight    = 0.2,
            smoothness_weight = 0.1,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.5,
        grad_window   = 5,
    ),

    # -- Stage 2: No forced pits, speed unlocked, agent must pit itself -------
    # Goal: full autonomous pit strategy at medium speed.
    # Forced pits are removed.  The agent must set pit_signal > 0 on its own.
    # Value function (bootstrapped) knows Q(s,pit) when tyre is worn.
    # Policy (zero-initialized pit row) has had ~200k steps of gradient signal.
    CurriculumStage(
        name                = "Stage 2 -- Speed, Agent Pits Autonomously",
        max_accel           = 11.0,
        forced_pit_interval = 0,     # agent alone decides pits from here
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.3,
        grad_window   = 5,
    ),

    # -- Stage 3: Full racing pace, autonomous pit strategy -------------------
    CurriculumStage(
        name                = "Stage 3 -- Full Racing + Pit Strategy",
        max_accel           = 15.0,
        forced_pit_interval = 0,
        reward_kwargs       = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 1.1,   # never graduates
        grad_window   = 9999,
    ),
]


# =============================================================================
# PIT STRATEGY V4 CURRICULUM  (d21)
# =============================================================================
#
# PROBLEM (d20 catastrophe — root cause):
#   All previous forced-pit schemes were TIME-BASED: force pit at step 50, 500.
#   This is unconditional — it fires regardless of tyre state.
#
#   d20 (interval=50): fired at step 50 → tyre_life ≈ 0.97 (nearly fresh).
#     → -200 pit penalty on fresh tyres → PPO learns "pitting always bad"
#     → pit_signal collapsed to -1.000 in first rollout
#     → driving destroyed (reward=-17.9, crashes at step 7)
#
#   Root insight: pit exploration must be STATE-CONDITIONAL.
#     Bad: "pit every 50 steps" → pits on fresh tyres → "pitting bad"
#     GOOD: "pit when tyre_life < 0.35" → pits on worn tyres → "pitting good when worn"
#
# THE D21 FIX — STATE-CONDITIONAL FORCED PITS:
#   forced_pit_threshold replaces forced_pit_interval for d21.
#   When tyre_life < threshold, env forces a pit (if not in cooldown).
#   Timing with ppo_tyre start (survives ~1500 steps):
#     tyre_life < 0.35 at ~step 929 — guaranteed to fire in every episode.
#     Forced pits ONLY happen on worn tyres → CORRECT training signal.
#     Value function learns: Q(s_worn, pit) >> Q(s_worn, no-pit).
#
# D21 ALSO:
#   - Starts from ppo_tyre weights (12D obs, 1500-step survival)
#     rather than from scratch or BC.
#   - load_ppo_tyre_into_ppo_pit() extends 2D action → 3D action:
#       action_net rows [0,1] from ppo_tyre (throttle/steer, preserved)
#       action_net row  [2] from bc_policy_pit_v3 (pit, weighted BC)
#   - REMOVES zero-init (d20 Fix B — catastrophic harm confirmed)
#   - KEEPS gamma=0.9999 (correct and necessary)
#
# STAGE SCHEDULE:
#   Stage 0: forced_pit_threshold=0.35 (~100k steps)
#     → Fires at ~step 929 in every episode. Value function bootstrapped.
#   Stage 1: forced_pit_threshold=0.25 (graduate at 50% lap rate)
#     → Agent must pit 0.25-0.35 on its own. Env is backup for 0-0.25.
#   Stage 2: forced_pit_threshold=0.0, max_accel=15.0 (graduate at 30%)
#     → Agent fully autonomous. Value function bootstrapped correctly.
#   Stage 3: forced_pit_threshold=0.0, full racing (never graduates)
#
STAGES_PIT_V4: List[CurriculumStage] = [

    # -- Stage 0: State-conditional forced pits at threshold=0.35 -----------
    # Goal: bootstrap value function with pit experiences ON WORN TYRES ONLY.
    #
    # WHY threshold=0.35:
    #   tyre_life wears at ~0.0007/step (base rate). Starting at 1.0:
    #   tyre_life < 0.35 at ~step 929.
    #   ppo_tyre survives ~1500+ steps → 100% of episodes reach step 929.
    #   Forced pits fire AT THE RIGHT STATE: worn tyres (~35% life left).
    #   No fresh-tyre pitting (tyre_life > 0.35 for steps 0-928).
    #   Value function learns: Q(s_tyre035, pit) >> Q(s_tyre035, no-pit).
    #
    # WHY max_accel=11.0 (not 6.0):
    #   We start from ppo_tyre which was trained at max_accel=11-15.
    #   Capping back to 6.0 would create unnecessary degradation.
    #   ppo_tyre already knows how to drive at 15 m/s.
    #
    # Graduation: grad_lap_rate=0.0 → any lap rate passes.
    # grad_window=50 → must see 50 rollouts (~102k steps) before graduating.
    CurriculumStage(
        name                 = "Stage 0 -- State-Conditional Pit Bootstrap (tyre<0.35)",
        max_accel            = 11.0,
        forced_pit_threshold = 0.35,  # force pit when tyre_life < 0.35
        forced_pit_interval  = 0,     # time-based forcing disabled
        reward_kwargs        = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.0,   # graduate after filling the window
        grad_window   = 50,    # 50 rollouts × 2048 steps ≈ 100k steps
    ),

    # -- Stage 1: Lower threshold=0.25 (backup for deeply worn tyres) -------
    # Goal: agent must handle tyre_life 0.25-0.35 autonomously.
    # The env only forces pits when tyre_life < 0.25 — a safety net.
    # The agent's own pit_signal must fire between tyre_life 0.25-0.35.
    # Value function (bootstrapped from Stage 0) should guide the agent.
    CurriculumStage(
        name                 = "Stage 1 -- Agent Pits 0.25-0.35, Env Backup <0.25",
        max_accel            = 11.0,
        forced_pit_threshold = 0.25,  # backup: force pit if tyre_life < 0.25
        forced_pit_interval  = 0,
        reward_kwargs        = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.5,
        grad_window   = 5,
    ),

    # -- Stage 2: No forced pits, full speed, agent fully autonomous --------
    # Goal: agent handles all pit decisions without assistance.
    # Forced pits disabled (threshold=0.0). Agent must set pit_signal > 0
    # when tyre_life is low (learned from Stages 0-1 value function).
    CurriculumStage(
        name                 = "Stage 2 -- Full Speed, Agent Pits Autonomously",
        max_accel            = 15.0,
        forced_pit_threshold = 0.0,   # no forced pits from here
        forced_pit_interval  = 0,
        reward_kwargs        = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.3,
        grad_window   = 5,
    ),

    # -- Stage 3: Full racing + autonomous pit strategy (never graduates) ---
    CurriculumStage(
        name                 = "Stage 3 -- Full Racing + Pit Strategy",
        max_accel            = 15.0,
        forced_pit_threshold = 0.0,
        forced_pit_interval  = 0,
        reward_kwargs        = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 1.1,   # never graduates
        grad_window   = 9999,
    ),
]


# =============================================================================
# CURRICULUM CALLBACK
# =============================================================================

class CurriculumCallback(BaseCallback):
    """
    SB3 callback that drives the curriculum stage progression.

    LIFECYCLE:
      _on_training_start():  applies Stage 1 to the live environment
      _on_step():            tracks crashes vs completions per episode
      _on_rollout_end():     computes rolling lap rate, checks graduation,
                             logs to TensorBoard

    HOW IT MODIFIES THE LIVE ENVIRONMENT:
      SB3's DummyVecEnv wraps the env in two layers:
        DummyVecEnv -> Monitor -> F1Env

      To reach the actual F1Env we call .unwrapped on the Monitor:
        inner_env = self.training_env.envs[0].unwrapped

      From there we directly assign:
        inner_env.car.max_accel  = new_accel
        inner_env.reward_fn      = RacingReward(**new_kwargs)

      This modifies the running environment IN PLACE. No reset, no new
      model. PPO sees different physics and reward from the next rollout.

      Interview version:
        "We modify environment parameters via the callback interface --
         a non-destructive live update that preserves the trained
         value function and policy weights."

    CRASH DETECTION:
      F1Env.step() adds "crashed": bool(terminated) to the info dict.
      When an episode ends, Monitor adds "episode" to the same dict.
      We check both: "episode" signals an episode just ended;
      info["crashed"] tells us HOW it ended.

        crashed=True  -- terminated (went off-track) -- bad
        crashed=False -- truncated  (survived 2000 steps) -- good
    """

    def __init__(self, stages: List[CurriculumStage] = STAGES, verbose: int = 1):
        super().__init__(verbose)
        self.stages    = stages
        self.stage_idx = 0   # start at Stage 1 (index 0)

        # Rolling window of per-rollout lap completion rates
        self._lap_window: List[float] = []

        # Per-rollout episode counters (reset each _on_rollout_end)
        self._ep_completed = 0   # episodes that survived to max_steps
        self._ep_total     = 0   # total episodes that finished this rollout

    # -- Convenience property -------------------------------------------------
    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    # -- Apply a stage to the live environment --------------------------------
    def _apply_stage(self, stage: CurriculumStage) -> None:
        """
        Push new physics + reward weights into the running F1Env.

        Navigation: DummyVecEnv -> envs[0] (Monitor) -> .unwrapped (F1Env)
        """
        from rl.rewards import RacingReward

        # Unwrap Monitor to get the bare F1Env
        inner_env = self.training_env.envs[0].unwrapped

        # Update the DynamicCar engine limit (controls terminal speed)
        inner_env.car.max_accel = stage.max_accel

        # Swap the reward function for this stage's weights
        inner_env.reward_fn = RacingReward(**stage.reward_kwargs)

        # d19: Update forced pit interval (0 = disabled, >0 = force pit every N steps).
        # hasattr check for backward compatibility with any env that predates this field.
        if hasattr(inner_env, 'forced_pit_interval'):
            inner_env.forced_pit_interval = stage.forced_pit_interval

        # d21: Update forced pit threshold (0.0 = disabled, >0 = force pit when
        # tyre_life < threshold). State-conditional, not time-conditional.
        if hasattr(inner_env, 'forced_pit_threshold'):
            inner_env.forced_pit_threshold = stage.forced_pit_threshold

        if self.verbose >= 1:
            print(f"\n[Curriculum] Applying: {stage.name}")
            print(f"             max_accel = {stage.max_accel} m/s^2")
            if stage.forced_pit_interval > 0:
                print(f"             forced_pit_interval = {stage.forced_pit_interval} steps")
            if stage.forced_pit_threshold > 0.0:
                print(f"             forced_pit_threshold = {stage.forced_pit_threshold:.2f} (tyre_life)")

    # -- Called once at training start ----------------------------------------
    def _on_training_start(self) -> None:
        """Activate Stage 1 immediately before the first rollout."""
        self._apply_stage(self.current_stage)

    # -- Called at every environment step -------------------------------------
    def _on_step(self) -> bool:
        """
        Track how each episode ends.

        self.locals["infos"] is a list of one dict per parallel env.
        "episode" appears in info when that episode just finished (added by Monitor).
        "crashed"  appears every step (added by F1Env.step()).

        We accumulate _ep_total and _ep_completed for the rollout-level
        calculation done in _on_rollout_end().
        """
        for info in self.locals["infos"]:
            if "episode" in info:
                # Episode just ended for this env
                self._ep_total += 1
                if not info.get("crashed", True):
                    # crashed=False = survived to max_steps = lap completed
                    self._ep_completed += 1

        return True   # returning False would abort training

    # -- Called at the end of each rollout ------------------------------------
    def _on_rollout_end(self) -> None:
        """
        Compute rolling lap rate, log to TensorBoard, check graduation.

        A "rollout" is n_steps steps (2048 by default). Multiple episodes
        may finish within one rollout, so _ep_total can be > 1.
        """
        stage = self.current_stage

        # Per-rollout lap completion rate
        rate = (self._ep_completed / self._ep_total) if self._ep_total > 0 else 0.0

        # Update rolling window -- keep only the last grad_window entries
        self._lap_window.append(rate)
        if len(self._lap_window) > stage.grad_window:
            self._lap_window.pop(0)

        rolling_rate = float(np.mean(self._lap_window))

        # Log to TensorBoard -- shows up under curriculum/ panel
        self.logger.record("curriculum/stage",        float(self.stage_idx))
        self.logger.record("curriculum/lap_rate",     rate)
        self.logger.record("curriculum/rolling_rate", rolling_rate)
        self.logger.record("curriculum/max_accel",    stage.max_accel)
        self.logger.record("curriculum/grad_target",  stage.grad_lap_rate)

        # Reset counters for the next rollout
        self._ep_completed = 0
        self._ep_total     = 0

        # -- Graduation check -------------------------------------------------
        # Conditions:
        #   1. Window is full (avoid graduating on a single lucky rollout)
        #   2. Rolling rate meets the stage's threshold
        #   3. There is a next stage to advance to
        next_idx    = self.stage_idx + 1
        window_full = len(self._lap_window) >= stage.grad_window

        if window_full and rolling_rate >= stage.grad_lap_rate and next_idx < len(self.stages):

            if self.verbose >= 1:
                print(f"\n[Curriculum] *** GRADUATED from {stage.name}! ***")
                print(f"             Rolling lap rate: {rolling_rate:.2f} "
                      f">= threshold {stage.grad_lap_rate:.2f}")
                print(f"             Advancing to: {self.stages[next_idx].name}\n")

            self.stage_idx   = next_idx
            self._lap_window = []   # reset window for fresh measurement of new stage
            self._apply_stage(self.stages[self.stage_idx])
