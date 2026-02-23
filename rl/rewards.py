"""
F1-Inspired Reward Shaping for the Racing Environment.

THEORY: WHY REWARD SHAPING IS NON-TRIVIAL
==========================================
Ng, Harada & Russell (1999) proved that you can add any "potential-based"
shaping term F(s,a,s') = γΦ(s') - Φ(s) without changing the optimal policy.
Any reward that doesn't have this form can alter what the agent considers
optimal — for better or worse.

Good reward design in racing:
  1. Primary signal = forward progress (everything else is secondary)
  2. Quadratic lateral penalty (weak near center, strong at edges)
  3. Input smoothness penalty (real F1: smooth inputs = less tyre slip)
  4. NO survival bonus (agent learns to survive, not to race)
  5. Large terminal penalty for off-track (sharp signal, fast learning)

OBSERVATION VECTOR (from f1_env.py):
  obs[0] = v / 20.0               (normalized speed, ~[0, 1])
  obs[1] = heading_error / π      (normalized heading, ~[-1, 1])
  obs[2] = lateral_error / 3.0   (normalized lateral, ~[-1, 1])
  obs[3] = sin(heading_error)     (smooth angle encoding)
  obs[4] = cos(heading_error)     (smooth angle encoding)
  obs[5] = curvature / π          (lookahead curvature, ~[-1, 1])

ACTION VECTOR:
  action[0] = throttle  ∈ [-1, 1]
  action[1] = steer     ∈ [-1, 1]
"""

import numpy as np


class RacingReward:
    """
    Modular reward function for the F1 racing environment.

    Separating reward logic from the environment has two benefits:
      1. You can unit-test the reward independently
      2. You can swap reward functions (e.g. for curriculum learning) without
         touching environment code

    Usage:
        reward_fn = RacingReward()
        reward = reward_fn.compute(obs, prev_obs, action, prev_action, terminated)
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
        Args:
            progress_weight:    Scale on v * cos(heading_error). This is the main
                                signal. Keep it at 1.0 as the baseline.

            speed_weight:       Small bonus for going fast on straights.
                                0.1 means max speed bonus is 0.1 per step.
                                Don't raise this too high or agent learns to speed
                                in corners (where crashing earns more speed bonus
                                than the lateral penalty costs).

            lateral_weight:     Coefficient on the quadratic lateral penalty.
                                0.5 * lateral² penalizes drift quadratically.
                                Quadratic is better than linear because:
                                  - Weak penalty near center (allow racing line freedom)
                                  - Strong penalty near track limits (avoid them)

            heading_weight:     Penalty for pointing away from track direction.
                                Keeps the car aligned during recovery maneuvers.

            smoothness_weight:  Penalty on change in steering/throttle between
                                consecutive steps. Encourages smooth inputs.
                                F1 reasoning: abrupt steering = tyre scrub = slower.
                                RL reasoning: smooth policies generalize better.

            terminal_penalty:   Large negative reward when car goes off-track.
                                This must be large enough to dominate several steps
                                of accumulated reward so the agent "fears" crashing.
                                Rule of thumb: > 5× the reward per step at max speed.
                                At max speed, reward ≈ 1.0/step. 20.0 > 5×1.0 ✓
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
        Compute the shaped reward for one transition (s, a, s').

        Args:
            obs:          Current observation (shape [6,])
            prev_obs:     Previous observation (shape [6,]), or None on first step
            action:       Current action [throttle, steer]
            prev_action:  Previous action, or None on first step
            terminated:   True if this step caused an off-track episode end

        Returns:
            reward: scalar float
        """

        # ── Unpack normalized observations ──────────────────────────────────
        v_norm       = obs[0]   # speed / 20.0,       range ≈ [0, 1]
        h_err_norm   = obs[1]   # heading_err / π,    range ≈ [-1, 1]
        lat_err_norm = obs[2]   # lateral_err / 3.0,  range ≈ [-1, 1]
        sin_h        = obs[3]   # sin(heading_error),  range [-1, 1]
        cos_h        = obs[4]   # cos(heading_error),  range [-1, 1]
        # obs[5] = curvature_norm, not used in reward but available

        # ── 1. PROGRESS REWARD ───────────────────────────────────────────────
        # v * cos(heading_error) = forward velocity projected onto track direction.
        #
        # Derivation:
        #   v_longitudinal = v * cos(heading_error)
        #   v_lateral      = v * sin(heading_error)
        # We want to reward v_longitudinal — speed along the track, not sideways.
        #
        # cos_h = cos(heading_error). Using the obs vector's cos directly is more
        # numerically stable than computing cos(h_err_norm * π).
        #
        # Range: v_norm ∈ [0,1], cos_h ∈ [-1,1]
        # Max value = 1.0 (full speed, perfectly aligned)
        # Negative when driving backwards (cos_h < 0) — good, penalizes that.
        progress = self.progress_weight * (v_norm * cos_h)

        # ── 2. SPEED BONUS ───────────────────────────────────────────────────
        # A small linear bonus for high speed. This encourages the agent to
        # build speed on straights where progress already rewards it, but adds
        # a secondary push when heading_error is large and progress drops.
        #
        # WARNING: Keep this small. A large speed weight causes the agent to
        # prioritize speed over track-following, leading to crashes.
        speed_bonus = self.speed_weight * v_norm

        # ── 3. LATERAL PENALTY (QUADRATIC) ──────────────────────────────────
        # Penalize distance from centerline.
        #
        # Why quadratic (x²) instead of linear (|x|)?
        #   |x| gives a constant gradient everywhere → agent treats 0.1m and
        #   0.9m off-center as equally bad per unit. Bad for racing lines.
        #
        #   x² gives gradient 2x → near center (x≈0) the gradient is tiny,
        #   giving the agent freedom to take the optimal racing line. Near
        #   limits (x≈1.0), gradient = 2.0, strongly pushing back to center.
        #   This is more F1-realistic (inside track limits = free choice).
        #
        # lat_err_norm ∈ [-1, 1] when within 3m of centerline.
        # Max penalty = 0.5 * 1.0² = 0.5 per step.
        lat_penalty = self.lateral_weight * (lat_err_norm ** 2)

        # ── 4. HEADING PENALTY ───────────────────────────────────────────────
        # Penalize heading misalignment (also quadratic for the same reason).
        # h_err_norm ∈ [-1, 1] where ±1 = perpendicular to track.
        # Max penalty = 0.1 * 1.0² = 0.1 per step.
        heading_penalty = self.heading_weight * (h_err_norm ** 2)

        # ── 5. SMOOTHNESS PENALTY ────────────────────────────────────────────
        # Penalize large changes in steering and throttle between consecutive steps.
        #
        # Physical motivation:
        #   dδ/dt (rate of steering change) causes tyre slip angle changes,
        #   which destabilize the vehicle at high speed. F1 drivers are trained
        #   to be smooth on all inputs.
        #
        # RL motivation:
        #   Smooth policies generalize better from simulation to reality (sim2real).
        #   Jerky policies exploit simulator artifacts that don't exist in the real world.
        #
        # Both actions ∈ [-1, 1], so delta ∈ [-2, 2].
        # steer changes weighted more (1.0) than throttle (0.5) because
        # sudden steering is more destabilizing than sudden acceleration.
        if prev_action is not None:
            steer_delta    = action[1] - prev_action[1]
            throttle_delta = action[0] - prev_action[0]
            smoothness_penalty = self.smoothness_weight * (
                steer_delta ** 2 + 0.5 * throttle_delta ** 2
            )
        else:
            # First step: no previous action to compare
            smoothness_penalty = 0.0

        # ── 6. TERMINAL PENALTY ──────────────────────────────────────────────
        # Large one-time penalty when the episode ends due to going off-track.
        # Must be large enough to override accumulated positive rewards.
        # See __init__ docstring for sizing rationale.
        term_penalty = self.terminal_penalty if terminated else 0.0

        # ── TOTAL REWARD ─────────────────────────────────────────────────────
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
        Returns a breakdown of each reward component for debugging/logging.
        Useful during training to monitor which terms dominate.

        Example use in TensorBoard:
            info = env.reward_fn.breakdown(obs, prev_obs, action, prev_action, done)
            writer.add_scalars("reward", info, global_step)
        """
        v_norm       = obs[0]
        h_err_norm   = obs[1]
        lat_err_norm = obs[2]
        cos_h        = obs[4]

        progress  = self.progress_weight * (v_norm * cos_h)
        speed     = self.speed_weight * v_norm
        lat       = -self.lateral_weight * (lat_err_norm ** 2)
        heading   = -self.heading_weight * (h_err_norm ** 2)

        if prev_action is not None:
            steer_delta    = action[1] - prev_action[1]
            throttle_delta = action[0] - prev_action[0]
            smooth = -self.smoothness_weight * (
                steer_delta ** 2 + 0.5 * throttle_delta ** 2
            )
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
