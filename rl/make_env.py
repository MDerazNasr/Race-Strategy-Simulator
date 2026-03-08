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


def make_env_multi_pit_d48b():
    """
    Creates a pit+multi-agent environment for SB3 (D48b).

    WHAT'S DIFFERENT vs make_env_multi_pit():
      - position_bonus lowered from 2.0 → 1.0 per step when ego is ahead.
      - D48 (position_bonus=2.0) achieved 76.9% dominance but crashed every episode
        (0% completion). The aggressive bonus over-incentivised risk-taking at the
        expense of track-limit awareness.
      - position_bonus=1.0: being ahead for a full 2000-step episode = +2000 reward.
        Still meaningfully above zero (D37 had no position bonus at all), but balanced
        against the driving quality terms so the agent preserves lap completion.

    STRATEGIC GOAL:
      Restore completion rate (D37=100%) while keeping the positional awareness
      that D48 proved can co-exist with pit timing (col[12]=0.023 > 0).
      Ideally: ≥1 pit, ≥10 laps, completion >50%, col[12]>0.

    TRAINING NOTE:
      Warm-start from D48 (ppo_multi_pit_d48.zip, 14D, 3D action, PitAwarePolicy).
      No obs extension needed — same 14D space.
      Recreate rollout buffer (clean state after load).
      Reset Adam optimizer (fresh momentum).
      3M steps (shorter — D48 already has combined features, just needs stability fix).

    Used in: rl/train_ppo_multi_pit_d48b.py (d48b)
    """
    from env.f1_multi_pit_env import F1MultiAgentPitEnv
    env = F1MultiAgentPitEnv(position_bonus=1.0)
    env = Monitor(env)
    return env


def make_env_monaco_d49(max_steps=6000, cache_dir='fastf1_cache', max_accel=8.0):
    """
    Creates a Monaco environment for no-curriculum PPO (D49).

    WHAT'S DIFFERENT vs make_env_monaco():
      - No CurriculumCallback — car.max_accel is fixed at 8.0 m/s² throughout.
      - D47 showed the STAGES curriculum is too aggressive for Monaco: Stage 0
        requires 50% lap completion, but Monaco (~3750m lap) is 12× longer than
        the oval so this threshold is never reached.
      - Fixed max_accel=8.0 m/s² gives the agent a stable speed ceiling (~20 m/s
        over a 10-second acceleration from rest) — moderate enough for Monaco's
        hairpins, high enough to be useful.
      - Same 11D obs, 2D action as make_env_monaco().

    STRATEGIC GOAL:
      Complete at least 1 Monaco lap, advancing beyond D42/D47's 0-lap result.
      Curvature weights should be non-zero (confirmed in D42/D47 even in failure).

    Used in: rl/train_ppo_monaco_d49.py (d49)
    """
    from env.track import load_fastf1_track
    from env.f1_env import F1Env
    track = load_fastf1_track(2023, 'Monaco', 'Q', n_points=300, cache_dir=cache_dir)
    env = F1Env(multi_lap=True, track=track, max_steps=max_steps)
    env.car.max_accel = max_accel
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


def make_env_monaco(max_steps=6000, cache_dir='fastf1_cache'):
    """
    Creates a Monaco (FastF1) environment instance for SB3 (D42).

    WHAT'S DIFFERENT vs make_env():
      - Real F1 circuit: Monaco 2023 qualifying fastest lap (~3248 m per lap).
      - F1Env(track=monaco_track, max_steps=6000, multi_lap=True).
      - Same 11D obs and 2D action as the standard env — no obs extension.
      - Track loaded via FastF1 telemetry; cached after first download.

    WHY MONACO?
      The oval track has constant curvature — the three lookahead observations
      [5]/[6]/[7] carry no information (every corner looks the same).
      Monaco has wildly varying geometry: long straight → tight hairpin →
      tunnel → chicane → swimming pool complex.  The curvature observations
      become genuinely useful: the agent must brake early for the hairpin and
      accelerate hard out of the tunnel.

    TRAINING NOTE:
      Train from scratch: expert data collection → BC warm start →
      3-stage curriculum PPO.  The oval cv2 policy cannot transfer — its
      weights were learned relative to the oval's coordinate system.

    EPISODE LENGTH:
      Monaco is ~3248 m. At 23 m/s average, 1 lap ≈ 141 s = 1410 steps.
      max_steps=6000 ≈ 4.3 laps — comparable to the oval's 15-lap episodes
      in terms of curriculum difficulty.

    Used in: rl/train_ppo_monaco.py (d42)
    """
    from env.track import load_fastf1_track
    track = load_fastf1_track(2023, 'Monaco', 'Q', n_points=300,
                               cache_dir=cache_dir)
    env = F1Env(multi_lap=True, track=track, max_steps=max_steps)
    env = Monitor(env)
    return env


def make_env_multi_agent_d41():
    """
    Creates a competitive multi-agent environment for SB3 (D41).

    WHAT'S DIFFERENT vs make_env_multi_agent():
      - F1MultiAgentEnv(opp_max_speed=27.0) — opponent matches ego's top speed.
      - D39 used opp_max_speed=22 m/s. The ego agent (cv2 lineage) cruises at
        ~26.9 m/s, so raw speed advantage was 4.9 m/s — trivial to win.
      - Raising the opponent to 27 m/s closes the shortcut: the ego can no
        longer simply drive fast and lap the opponent.  It must use track_gap
        and positional strategy to stay ahead or overtake.

    STRATEGIC GOAL:
      Force non-zero weight on dims 11 (track_gap) and 12 (opp_speed_norm).
      In D39, both had learned weight = 0 because speed alone was sufficient.
      With an equally-fast opponent, positional awareness becomes necessary.

    TRAINING NOTE:
      Warm-start from D39 (ppo_multi_agent_d39.zip, 13D obs, 2D action).
      No obs extension needed — same 13D obs space, same action space.
      Recreate rollout buffer (clean state after load).
      ent_coef=0.01 from start — prevents log_std collapse (d38 lesson).

    Used in: rl/train_ppo_multi_agent_d41.py (d41)
    """
    from env.f1_multi_env import F1MultiAgentEnv
    env = F1MultiAgentEnv(opp_max_speed=27.0)
    env = Monitor(env)
    return env


def make_env_multi_agent_d43():
    """
    Creates a competitive multi-agent environment for SB3 (D43).

    WHAT'S DIFFERENT vs make_env_multi_agent_d41():
      - F1MultiAgentEnv(opp_max_speed=25.0) — opponent is fast but not equal.
      - D41 used opp_max_speed=27.0 (ego=27 m/s) → training reward 3170 but
        deterministic eval = 0 laps, 0% completion. Equal-speed competition
        is too hard: no speed buffer means any misalignment causes collision,
        and the policy relies entirely on stochastic noise to survive.
      - 25 m/s closes the raw-speed shortcut (D39 opp=22: ego 26.9 m/s had
        4.9 m/s advantage → no positional awareness needed) while still giving
        the ego a ~2 m/s buffer (~8% speed advantage) to complete laps reliably.

    STRATEGIC GOAL:
      Prove simultaneous: (1) track_gap weight non-zero AND (2) reliable lap
      completion. Neither D39 (weight=0, laps=17) nor D41 (weight>0, laps=0)
      achieved both at once. D43 is the sweet spot.

    TRAINING NOTE:
      Warm-start from D41 (ppo_multi_agent_d41.zip, 13D obs, 2D action).
      D41 already has non-zero track_gap weights from learning against the 27
      m/s opponent — it should adapt faster to the slightly easier 25 m/s task.
      No obs extension needed — same 13D obs space, same action space.
      Reset Adam optimizer (D41 optimizer state may have stale momentum from
      its own convergence at 27 m/s — fresh optimizer avoids interference).

    Used in: rl/train_ppo_multi_agent_d43.py (d43)
    """
    from env.f1_multi_env import F1MultiAgentEnv
    env = F1MultiAgentEnv(opp_max_speed=25.0)
    env = Monitor(env)
    return env


def make_env_multi_agent_d44():
    """
    Creates a competitive multi-agent environment for SB3 (D44).

    WHAT'S DIFFERENT vs make_env_multi_agent_d43():
      - position_bonus raised from 0.5 → 2.0 per step when ego is ahead.
      - Everything else identical: opp_max_speed=25.0, overtake_bonus=200,
        collision_radius=3.0m, collision_penalty=0.5/step.

    WHY D43 FAILED (deterministic eval = 0 laps, 275 reward):
      D43's policy converged to a "follow closely" equilibrium.
      At opp=25 m/s, the deterministic mean throttle yielded ego speed 24.36 m/s
      — BELOW the opponent's 25 m/s. The agent was frequently behind the opponent,
      accumulating only 0.5/step when occasionally ahead. The follow strategy
      (no collision, small position bonus) was a local optimum.

    WHY position_bonus=2.0 FIXES THIS:
      Being ahead for a full 2000-step episode now yields +4000 vs the old +1000.
      The reward gradient strongly favors "drive fast and stay ahead" over "follow."
      The overtake must still be executed, but the RETURN on being ahead is 4x,
      making it worth the effort of pushing through the opponent's position.

      Value of being ahead:    0.5 * 2000 = +1000 (old)  →  2.0 * 2000 = +4000 (new)
      Value of following behind:  0 (unchanged)
      Net advantage of overtaking: +1000 → +4000

    TRAINING NOTE:
      Warm-start from D43 (ppo_multi_agent_d43.zip, 13D obs, 2D action).
      D43 already has strong track_gap weights (0.022) — it knows how to read
      position, just needs the incentive to act on it more aggressively.

    Used in: rl/train_ppo_multi_agent_d44.py (d44)
    """
    from env.f1_multi_env import F1MultiAgentEnv
    env = F1MultiAgentEnv(opp_max_speed=25.0, position_bonus=2.0)
    env = Monitor(env)
    return env


def make_env_multi_pit():
    """
    Creates a pit+multi-agent environment instance for SB3 (D48).

    WHAT'S DIFFERENT vs make_env_pit_d30() and make_env_multi_agent():
      - F1MultiAgentPitEnv: ego with tyre degradation + pit stops vs ExpertDriver.
      - Observation is 14D: [11D standard] + [tyre_life] + [track_gap] + [opp_speed_norm].
        tyre_life is at dim 11 — same index as D32–D37 — so PitAwarePolicy works unchanged.
      - Action space is 3D: [throttle, steer, pit_signal] (same as D37).
      - Opponent: ExpertDriver at 25 m/s, no tyre degradation.
      - Additional reward terms vs make_env_pit_d30():
          position_bonus   = +2.0/step when ego ahead (proven in D44/D46)
          overtake_bonus   = +200 one-time on overtake
          collision_penalty= -0.5/step within 3m
      - Pit rewards (same as D37 / make_env_pit_d30):
          pit_penalty      = -200
          voluntary_pit_bonus = +300 when tyre_life < 0.60 and agent pits voluntarily

    STRATEGIC GOAL:
      Agent must learn BOTH pit timing (D37 lesson: pit every ~600 steps)
      AND positional strategy (D46 lesson: actively use track_gap to stay ahead).
      Ideally: undercut — pit early relative to opponent's pace to take position.

    TRAINING NOTE:
      Warm-start from D37 (ppo_pit_v4_d37.zip, 12D, 3D action, PitAwarePolicy).
      extend_obs_dim(model, 12, 14) zero-pads dims 12–13.
      PitAwarePolicy.TYRE_LIFE_OBS_IDX = 11 → reads tyre_life correctly.
      Fix obs bounds: dim 12 (track_gap) low=-1.0.
      Recreate rollout buffer after extension.

    Used in: rl/train_ppo_multi_pit.py (d48)
    """
    from env.f1_multi_pit_env import F1MultiAgentPitEnv
    env = F1MultiAgentPitEnv()
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
