"""
F1-Inspired Reward Shaping for the Racing Environment.

WHY REWARD ENGINEERING MATTERS
================================
In RL, reward = the ONLY signal the agent has about what "good" means.
The agent doesn't understand "lap time" or "racing line" — it only knows:
  "actions that led to higher reward → do more of that"
  "actions that led to lower reward → do less of that"

Every behavior you want must be encoded in the reward.
Behaviors NOT in the reward will be ignored or exploited.

THEORY: POTENTIAL-BASED SHAPING (Ng, Harada & Russell 1999)
=============================================================
You can add any function F(s,a,s') = gamma*Phi(s') - Phi(s) to your
reward without changing the OPTIMAL POLICY. This is "potential-based shaping."

Reward shaping that is NOT potential-based CAN change what the optimal
policy is — sometimes for the better, sometimes accidentally for the worse.

Our reward is not strictly potential-based, but each term is carefully
sized so the dominant term (progress) aligns with the actual task objective.

OBSERVATION VECTOR (from env/f1_env.py):
  obs[0] = v / 20.0               normalized speed        range ~ [0, 1]
  obs[1] = heading_error / pi     normalized heading      range ~ [-1, 1]
  obs[2] = lateral_error / 3.0   normalized lateral      range ~ [-1, 1]
  obs[3] = sin(heading_error)     smooth angle encoding   range   [-1, 1]
  obs[4] = cos(heading_error)     smooth angle encoding   range   [-1, 1]
  obs[5] = curvature / pi         lookahead curvature     range ~ [-1, 1]

ACTION VECTOR:
  action[0] = throttle  in [-1, 1]
  action[1] = steer     in [-1, 1]
"""

import numpy as np


class RacingReward:
    """
    Modular reward function for the F1 racing environment.

    Separating reward logic from the environment has two benefits:
      1. You can unit-test the reward independently of physics
      2. You can swap reward functions for curriculum learning (Week 4)
         without touching environment code

    Usage:
        reward_fn = RacingReward()
        r = reward_fn.compute(obs, prev_obs, action, prev_action, terminated)
    """

    def __init__(
        self,
        progress_weight: float = 1.0,
        speed_weight: float = 0.1,
        lateral_weight: float = 0.5,
        heading_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        terminal_penalty: float = 20.0,
    ):
        """
        Store all reward weights as instance attributes.

        Why store as attributes instead of hardcoding?
          - Makes it easy to run hyperparameter sweeps: RacingReward(lateral_weight=0.8)
          - Easy to log which reward config was used for a given training run
          - Supports curriculum learning: swap reward_fn mid-training (Week 4)

        Args:
            progress_weight:    Scale on v * cos(heading_error).
                                The PRIMARY signal. Keep at 1.0 as the reference.
                                Everything else should be smaller than this.

            speed_weight:       Small bonus for going fast (0.1 = max 10% of primary).
                                Keep SMALL or agent learns to speed into walls.

            lateral_weight:     Quadratic penalty for lateral drift (0.5 * lat^2).
                                Quadratic = weak near center, strong near limits.
                                Gives freedom to take the racing line.

            heading_weight:     Quadratic penalty for heading misalignment.
                                Keeps car pointed in the right direction.

            smoothness_weight:  Penalty on input rate-of-change between steps.
                                F1 reasoning: smooth inputs = less tyre scrub.
                                RL reasoning: smooth policies generalize better.

            terminal_penalty:   One-time penalty for going off track.
                                Must dominate several steps of positive reward.
                                Rule of thumb: > 5 * max_reward_per_step.
                                Max per-step ~ 1.1, so 20.0 > 5*1.1. Good.
        """
        self.progress_weight = progress_weight
        self.speed_weight = speed_weight
        self.lateral_weight = lateral_weight
        self.heading_weight = heading_weight
        self.smoothness_weight = smoothness_weight
        self.terminal_penalty = terminal_penalty

    def compute(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        terminated: bool,
    ) -> float:
        """
        Compute the shaped reward for one transition (state, action, next_state).

        Args:
            obs:         Current observation  shape (6,)
            prev_obs:    Previous observation shape (6,), or None on first step
            action:      Current action [throttle, steer]
            prev_action: Previous action,   or None on first step
            terminated:  True if this step caused an off-track episode end

        Returns:
            reward: scalar float
        """

        # ── Unpack normalized observations ───────────────────────────────────
        v_norm       = obs[0]   # speed / 20.0
        h_err_norm   = obs[1]   # heading_error / pi
        lat_err_norm = obs[2]   # lateral_error / 3.0
        cos_h        = obs[4]   # cos(heading_error)

        # ── 1. PROGRESS ──────────────────────────────────────────────────────
        # v * cos(heading_error) = forward velocity projected onto track direction.
        #
        # Think of it as a dot product:
        #   v_forward = v * cos(angle_between_car_and_track)
        #   v_lateral = v * sin(angle_between_car_and_track)
        #
        # We want to reward v_forward (along the track), not v_lateral (sideways).
        # Using cos_h from obs[4] is more stable than recomputing cos(h_err_norm*pi).
        #
        # Range: v_norm in [0,1], cos_h in [-1,1]
        # At max speed, aligned: 1.0 * 1.0 = 1.0 per step
        # Driving backwards: negative (agent is penalized for reversing)
        progress = self.progress_weight * (v_norm * cos_h)

        # ── 2. SPEED BONUS ───────────────────────────────────────────────────
        # Small additional incentive for speed regardless of direction.
        # Prevents the agent from going slow-but-aligned as a safe strategy.
        # Max value = 0.1 per step (10% of max progress signal).
        speed_bonus = self.speed_weight * v_norm

        # ── 3. LATERAL PENALTY (QUADRATIC) ───────────────────────────────────
        # WHY QUADRATIC (x^2) NOT LINEAR (|x|)?
        #
        # Linear |x|: gradient = ±1 everywhere.
        #   The penalty per meter of drift is CONSTANT.
        #   Agent treats being 0.1m off-center as equally bad per unit as 2.9m.
        #   This is wrong for racing — near center, freedom to maneuver matters.
        #
        # Quadratic x^2: gradient = 2x.
        #   At x=0.1 (0.3m real): gradient=0.2, weak signal → freedom to race
        #   At x=0.9 (2.7m real): gradient=1.8, strong signal → get back now
        #   At x=1.0 (3.0m real): gradient=2.0 → right before termination
        #
        # Max penalty = 0.5 * 1.0^2 = 0.5 per step (50% of max progress).
        lat_penalty = self.lateral_weight * (lat_err_norm ** 2)

        # ── 4. HEADING PENALTY (QUADRATIC) ───────────────────────────────────
        # Penalizes facing the wrong direction.
        # Quadratic for the same reason as lateral — freedom near-aligned.
        # Max penalty = 0.1 * 1.0^2 = 0.1 per step (10% of max progress).
        heading_penalty = self.heading_weight * (h_err_norm ** 2)

        # ── 5. SMOOTHNESS PENALTY ─────────────────────────────────────────────
        # Penalizes large CHANGES in steering and throttle between consecutive steps.
        #
        # Physical motivation (F1):
        #   dδ/dt (steering rate) causes tyre slip angle changes.
        #   At high speed, abrupt steering destabilizes the car.
        #   Real F1 drivers are trained to be smooth on all inputs.
        #
        # RL motivation:
        #   Smooth policies generalize better sim-to-real.
        #   Jerky policies exploit simulation artifacts (perfect traction, no lag)
        #   that don't exist on real hardware.
        #
        # action[1] = steer, action[0] = throttle
        # Steer weighted 1.0, throttle weighted 0.5 (steering more destabilizing)
        # Both are in [-1, 1], so delta in [-2, 2], penalty max = 0.05 * 4 = 0.20
        if prev_action is not None:
            steer_delta    = action[1] - prev_action[1]
            throttle_delta = action[0] - prev_action[0]
            smoothness_penalty = self.smoothness_weight * (
                steer_delta ** 2 + 0.5 * throttle_delta ** 2
            )
        else:
            # First step: no previous action to compare against
            smoothness_penalty = 0.0

        # ── 6. TERMINAL PENALTY ──────────────────────────────────────────────
        # Large one-time hit when the episode ends from going off-track.
        # This teaches the agent: "crossing track limits is catastrophic."
        # Without a large terminal penalty, the agent may accept occasional
        # crashes because the accumulated positive reward from speed outweighs
        # the small per-step penalties.
        term_penalty = self.terminal_penalty if terminated else 0.0

        # ── TOTAL ─────────────────────────────────────────────────────────────
        reward = (
            progress
            + speed_bonus
            - lat_penalty
            - heading_penalty
            - smoothness_penalty
            - term_penalty
        )

        return float(reward)

    def breakdown(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        terminated: bool,
    ) -> dict:
        """
        Returns a named breakdown of each reward component.

        Use this during debugging or TensorBoard logging to see WHICH
        terms are dominating the reward. Common issues:
          - lateral term dominates → agent learned to brake to avoid drift
          - smoothness dominates   → agent learned to be slow and static
          - progress dominates     → agent is genuinely racing (desired!)

        Example:
            bd = env.reward_fn.breakdown(obs, prev_obs, action, prev_action, done)
            # {'progress': 0.87, 'speed': 0.05, 'lateral': -0.02, ...}
        """
        v_norm       = obs[0]
        h_err_norm   = obs[1]
        lat_err_norm = obs[2]
        cos_h        = obs[4]

        progress = self.progress_weight * (v_norm * cos_h)
        speed    = self.speed_weight * v_norm
        lat      = -self.lateral_weight * (lat_err_norm ** 2)
        heading  = -self.heading_weight * (h_err_norm ** 2)

        if prev_action is not None:
            sd = action[1] - prev_action[1]
            td = action[0] - prev_action[0]
            smooth = -self.smoothness_weight * (sd ** 2 + 0.5 * td ** 2)
        else:
            smooth = 0.0

        terminal = -self.terminal_penalty if terminated else 0.0

        return {
            "progress":   progress,
            "speed":      speed,
            "lateral":    lat,
            "heading":    heading,
            "smoothness": smooth,
            "terminal":   terminal,
            "total":      progress + speed + lat + heading + smooth + terminal,
        }
