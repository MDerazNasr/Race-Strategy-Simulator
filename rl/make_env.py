"""
what is SB3?
- SB3 is a reinforcement learning library that provides a
set of tools for training and evaluating reinforcement learning
algorithms.
"""

import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env


def make_env():
    """
    Creates a fresh environment instance for SB3.
    Monitor tracks episode reward/length so SB3 can plot learning curves.

    Standard mode: episodes end after 2000 steps OR on crash.
    Used for all training up to and including d15 (ppo_curriculum_v2).
    """
    env = F1Env()
    env = Monitor(env)
    return env


def make_env_multi_lap():
    """
    Creates a multi-lap environment instance for SB3.
    Episodes ONLY end on crash — no 2000-step truncation.

    WHY A SEPARATE FACTORY?
      SB3's DummyVecEnv takes a list of callables: [make_env, make_env, ...]
      Each callable must be a zero-argument function.
      Having two named factories (make_env and make_env_multi_lap) lets the
      training scripts select the mode without any lambda gymnastics.

    Used in: rl/train_ppo_multi_lap.py (d16+)
    NOT used in: evaluate.py — evaluation keeps fixed 2000-step episodes
      so that all policies are compared on equal terms (same episode length).
    """
    env = F1Env(multi_lap=True)
    env = Monitor(env)
    return env


def make_env_tyre():
    """
    Creates a tyre-degradation environment instance for SB3 (Week 5 / d17+).

    WHAT'S DIFFERENT vs make_env():
      - F1Env(tyre_degradation=True) is used.
      - Observation vector is 12D (appends tyre_life ∈ [0, 1]).
      - DynamicCar.tyre_life degrades every step based on slip angles.
        Grip = mu_base * max(0.1, tyre_life).
      - Episodes still end at max_steps=2000 (no multi_lap).
        This is intentional: avoids the catastrophic forgetting of d16.

    WHY 12D OBS?
      Without tyre_life in the observation, the agent is blind to its tyre
      state.  It cannot develop strategy because it cannot distinguish
      "I have 80% tyre left" from "I have 20% tyre left."
      Adding obs[11] = tyre_life gives the agent the information it needs
      to learn when to conserve and when to push.

    OBSERVATION SPACE:
      [0-10]  Same as standard make_env() (speed, heading, lateral, ...)
      [11]    tyre_life (normalised, 1.0 = new, 0.0 = fully worn)

    Used in: rl/train_ppo_tyre.py (d17+)
    NOT used in: evaluate.py for old policies (they expect 11D obs).
      Tyre-trained policies are evaluated with a tyre env wrapper.
    """
    env = F1Env(tyre_degradation=True)
    env = Monitor(env)
    return env


def make_env_pit():
    """
    Creates a pit-stop environment instance for SB3 (Week 5 / d18+).

    WHAT'S DIFFERENT vs make_env_tyre():
      - F1Env(tyre_degradation=True, pit_stops=True) is used.
      - Action space is 3D: [throttle, steer, pit_signal].
        pit_signal > 0 → request a pit stop this step.
      - Observation is still 12D (same as make_env_tyre).
        The agent uses obs[11] = tyre_life to decide when to pit.
      - Pitting costs PIT_PENALTY = -200 reward but resets tyre_life to 1.0.
      - pit_cooldown = 100 steps prevents immediate re-pitting.

    WHY TRAIN FROM SCRATCH (not continue from ppo_tyre)?
      The action space changed: 2D → 3D.  PPO.load() reconstructs the
      policy's action_net as Linear(128, 2).  There is no zero-padding trick
      for action outputs — the new pit_signal output layer has no prior
      weights to inherit.  We must build a new PPO with a 3D action_net.

      The warm start comes from BC instead: expert demonstrations are
      collected with include_pit=True, a BC policy is trained on 12D→3D,
      and that policy initialises the PPO actor (same pipeline as d13).

    OBSERVATION SPACE:  [0-11] same as make_env_tyre
    ACTION SPACE:       [throttle, steer, pit_signal]

    Used in: rl/train_ppo_pit.py (d18+)
    """
    env = F1Env(tyre_degradation=True, pit_stops=True)
    env = Monitor(env)
    return env


def make_env_pit_d30():
    """
    Creates a pit-stop environment with voluntary pit reward shaping (Week 5 / d30).

    WHAT'S DIFFERENT vs make_env_pit():
      - F1Env(tyre_degradation=True, pit_stops=True, voluntary_pit_reward=True)
      - Voluntary pit reward: +300 bonus when agent's OWN pit_signal > 0 fires
        AND tyre_life < 0.60 (worn-tyre zone). No forced pit at all.

    WHY VOLUNTARY PIT REWARD? (d26–d29 lesson):
      Forced pits (d21–d29) fire regardless of the agent's pit_signal. This means
      the advantage from a forced pit flows back to whatever action the agent
      happened to choose — usually pit_signal < 0 (negative bias). The gradient
      then reinforces pit_signal < 0: bias went +0.006 → -0.817 → -1.217 across
      d26/d27/d28/d29. Wrong direction every time.

      voluntary_pit_reward fires ONLY when pit_signal > 0 (agent's choice).
      Net cost: -200 (penalty) + 300 (bonus) = +100 → immediate profit from pitting.
      Plus: fresh tyres → survive to step 2000 → ~+800 more reward.
      PPO sees: (worn_tyre_state, pit_signal > 0) → high reward → correct gradient.

    THRESHOLD 0.60 (reachable from fixed-start trajectory):
      tyre_life reaches 0.60 at step ~571 on fixed-start — well before the
      crash at step 1354. Agent can voluntarily pit from step 571 onwards.
      D21 pit_std ≈ 2.13 (log_std=0.76), bias ≈ +0.006 → P(pit>0) ≈ 50%.
      → Agent discovers the voluntary pit bonus within the first few episodes.

    Used in: rl/train_ppo_pit_v4_d30.py (d30)
    """
    env = F1Env(tyre_degradation=True, pit_stops=True, voluntary_pit_reward=True)
    env = Monitor(env)
    return env


def make_env_sc():
    """
    Creates a safety car environment instance for SB3 (Week 6 / d40).

    WHAT'S DIFFERENT vs make_env_pit_d30():
      - F1Env(..., safety_car=True) is used.
      - Observation is 13D: [12D pit env obs] + [sc_active ∈ {0, 1}].
      - Action space is 3D (same as pit env): [throttle, steer, pit_signal].
      - Additional reward terms:
          sc_speed_penalty  = -2.0 × max(0, v - 22.0) per step during SC
          sc_pit_bonus      = +100 when agent pits under SC (net cost: -100 vs -200)

    STRATEGIC GOAL:
      Agent must learn: when sc_active=1, slow to ~22 m/s AND consider pitting
      (pit costs only -100 under SC vs -200 outside). This is the "undercut" —
      pit under safety car to minimize lap time lost relative to opponents.

    SC PARAMETERS (see f1_env.py safety car block for rationale):
      sc_trigger_prob = 0.003   (~1 SC per 2.5 laps)
      sc_speed_limit  = 22.0    (m/s, ~80 km/h)
      sc_duration     = 80–200  (steps = 8–20 seconds)
      sc_cooldown     = 300     (steps between SCs)
      sc_speed_penalty = 2.0    (per m/s over limit)

    TRAINING NOTE:
      Load ppo_pit_v4_d37 (12D, 3D action) and extend to 13D using
      extend_obs_dim(model, 12, 13). sc_active bounds [0, 1] are
      the extend_obs_dim defaults — no manual fix needed.
      Recreate rollout buffer after extension (same fix as D39).

    Used in: rl/train_ppo_sc_d40.py (d40)
    """
    env = F1Env(
        multi_lap=True,
        tyre_degradation=True,
        pit_stops=True,
        voluntary_pit_reward=True,
        voluntary_pit_threshold=0.60,
        safety_car=True,
    )
    env = Monitor(env)
    return env


def make_env_multi_agent():
    """
    Creates a multi-agent environment instance for SB3 (D39).

    WHAT'S DIFFERENT vs make_env():
      - F1MultiAgentEnv: ego PPO vs ExpertDriver opponent (max_speed=22 m/s).
      - Observation is 13D: [11D standard ego obs] + [track_gap] + [opp_speed_norm].
      - Action space is 2D (same as make_env): no pit signal.
      - Additional reward terms:
          position_bonus   = +0.5/step when ego is ahead (track_gap < 0)
          overtake_bonus   = +200 one-time when ego overtakes opponent
          collision_penalty= -0.5/step when distance < 3m
      - No tyre degradation (focus is on race strategy vs opponent).

    TRAINING NOTE:
      Load ppo_curriculum_v2 (11D) and extend to 13D using extend_obs_dim(model, 11, 13).
      After extend_obs_dim, fix obs dim 11 bounds to [-1, 1] for track_gap.
      Use ent_coef=0.01 from the start to prevent log_std collapse (lesson from d38).

    Used in: rl/train_ppo_multi_agent_d39.py (d39)
    """
    from env.f1_multi_env import F1MultiAgentEnv
    env = F1MultiAgentEnv()
    env = Monitor(env)
    return env


def make_env_pit_d23():
    """
    Creates a pit-stop environment with pit timing reward shaping (Week 5 / d23).

    WHAT'S DIFFERENT vs make_env_pit():
      - F1Env(tyre_degradation=True, pit_stops=True, pit_timing_reward=True)
      - Pit timing reward shaping is enabled:
          tyre_life < 0.3 when agent pits: +100 bonus (net cost = -100)
          tyre_life > 0.5 when agent pits: -100 extra penalty (net cost = -300)
          Neutral zone (0.3–0.5): no change (net cost = -200)

    WHY PIT TIMING REWARD? (d22 lesson):
      d22 showed that the -200 PIT_PENALTY alone is not enough to maintain
      pitting behavior during fine-tuning. The agent found it more profitable
      to drive perfectly and skip pitting entirely (+406 reward gain from
      better driving > pit benefit). Without explicit pit timing incentive,
      the agent rationally chose "never pit."

      pit_timing_reward adds:
        1. A POSITIVE signal for pitting at the right time (worn tyres)
        2. A NEGATIVE signal for pitting at the wrong time (fresh tyres)
      Together they make correct pit timing explicitly rewarded.

    Used in: rl/train_ppo_pit_v4_d23.py (d23)
    NOT used in: evaluate.py — evaluation uses standard env_pit (no timing bonus)
      so that reward numbers are comparable across d21/d22/d23.
    """
    env = F1Env(tyre_degradation=True, pit_stops=True, pit_timing_reward=True)
    env = Monitor(env)
    return env
