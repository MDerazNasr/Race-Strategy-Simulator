"""
F1MultiAgentPitEnv — Ego PPO (with tyre degradation + pit stops) vs ExpertDriver (D48).

WHY D48?
=========
D37 proved the agent can learn strategic pit timing (3 voluntary pits, 23.67 m/s).
D46 proved the agent can learn positional strategy against an opponent (7943 reward,
col[11] track_gap weight 0.000 → 0.070).

D48 combines BOTH: the ego must simultaneously pit at the right time AND beat the
opponent. No experiment has done this before. Key questions:

  1. Does pit knowledge (D37) survive the addition of an opponent signal?
  2. Does positional awareness (D46) survive the constraint of tyre management?
  3. Is there an "undercut" strategy: pit earlier than scheduled to take position?

OBSERVATION (14D):
==================
  [0-10]  Standard 11D ego obs (speed, heading, lateral, curvature, progress, v_y, r)
  [11]    tyre_life ∈ [0, 1]       ← CRITICAL: obs[11] = tyre_life for PitAwarePolicy
  [12]    track_gap ∈ [-1, 1]      ← opponent_progress − ego_progress, wrap-corrected
            Positive = opponent ahead. Negative = ego ahead.
  [13]    opp_speed_norm ∈ [0, 1]  ← opp_car.v / 30.0

WHY tyre_life AT DIM 11:
  PitAwarePolicy.TyrLifeAugmentedExtractor uses TYRE_LIFE_OBS_IDX = 11 to read
  tyre_life directly from the observation and append it to the actor's 128-dim latent.
  Placing tyre_life at dim 11 (same as D32–D37's 12D layout) ensures PitAwarePolicy
  works unchanged when warm-starting from D37.

  The multi-agent dims (track_gap, opp_speed_norm) are at 12–13, beyond what
  PitAwarePolicy inspects. They learn via the standard MLP policy_net[0] columns.

ACTION (3D):
============
  [throttle, steer, pit_signal]  (same as D37)
  pit_signal > 0 fires a pit stop if cooldown = 0.

DESIGN:
=======
  - Ego: DynamicCar(base_wear=0.0003, slip_coef=0.002) — same tyre parameters as D37.
  - Opponent: DynamicCar() — no tyre degradation; ExpertDriver at opp_max_speed=25 m/s.
  - Rewards: RacingReward base + position_bonus(2.0) + overtake_bonus(200) +
             lap_bonus(100) + pit_penalty(-200) + voluntary_pit_bonus(+300 at tl<0.60)
             − collision_penalty(0.5/step within 3m)
  - Termination: |lateral_error| > 3.0 m; truncation at 2000 steps.

WARM-START (from D37):
  D37 = 12D obs, 3D action, PitAwarePolicy.
  extend_obs_dim(model, 12, 14) zero-pads dims 12–13 → PitAwarePolicy unchanged.
  New columns start silent and learn to signal track_gap and opp_speed over training.
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.car_model import DynamicCar
from env.track import generate_oval_track, closest_point
from utils.geometry import normalize_angle, track_tangent, signed_lateral_error
from expert.expert_driver import ExpertDriver
from rl.rewards import RacingReward


class F1MultiAgentPitEnv(gym.Env):
    """
    Ego PPO car (with tyre degradation + pit stops) vs ExpertDriver opponent.

    Observation: 14D = [11D standard ego obs] + [tyre_life] + [track_gap] + [opp_speed_norm]
    Action:      3D  = [throttle, steer, pit_signal]
    """

    metadata = {"render_modes": [], "render_fps": 30}

    # Pit stop constants (same as F1Env)
    PIT_PENALTY          = -200.0
    PIT_COOLDOWN_STEPS   = 100
    VOLUNTARY_PIT_BONUS  = 300.0   # net cost = -200 + 300 = +100
    VOLUNTARY_PIT_THRESH = 0.60    # bonus fires when tyre_life < 0.60

    def __init__(
        self,
        dt: float = 0.1,
        opp_max_speed: float = 25.0,
        position_bonus: float = 2.0,
        overtake_bonus: float = 200.0,
        overtake_cooldown: int = 200,
        collision_radius: float = 3.0,
        collision_penalty: float = 0.5,
    ):
        super().__init__()
        self.dt = dt

        # ── Track ──────────────────────────────────────────────────────────────
        self.track  = generate_oval_track()
        self.n_wpts = len(self.track)

        # ── Cars ───────────────────────────────────────────────────────────────
        # Ego: DynamicCar with tyre degradation (same params as F1Env D32–D37)
        self.car = DynamicCar(base_wear=0.0003, slip_coef=0.002)
        # Opponent: DynamicCar with no degradation — fixed speed expert
        self.opp_car = DynamicCar()

        # ── Opponent policy ────────────────────────────────────────────────────
        self.expert = ExpertDriver(
            self.track,
            max_speed=opp_max_speed,
            lookahead=8,
            corner_factor=12.0,
            include_pit=False,
        )

        # ── Racing reward (ego only) ───────────────────────────────────────────
        self.reward_fn = RacingReward()

        # ── Action space: [throttle, steer, pit_signal] ────────────────────────
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0], dtype=np.float32),
        )

        # ── Observation space: 14D ─────────────────────────────────────────────
        # [0-10]  standard 11D (same bounds as F1Env/F1MultiAgentEnv)
        # [11]    tyre_life ∈ [0, 1]
        # [12]    track_gap ∈ [-1, 1]
        # [13]    opp_speed_norm ∈ [0, 1]
        obs_high = np.array(
            [100.0, np.pi, 50.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0, 20.0, 5.0,
             1.0,   # tyre_life
             1.0,   # track_gap
             1.0],  # opp_speed_norm
            dtype=np.float32,
        )
        obs_low = np.array(
            [0.0, -np.pi, -50.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0, -20.0, -5.0,
             0.0,   # tyre_life
             -1.0,  # track_gap
             0.0],  # opp_speed_norm
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ── Episode parameters ──────────────────────────────────────────────────
        self.max_steps = 2000
        self.step_count = 0
        self.lap_bonus  = 100.0

        # ── Competitive parameters ──────────────────────────────────────────────
        self.position_bonus        = position_bonus     # +2.0/step when ego ahead
        self.overtake_bonus        = overtake_bonus     # +200 one-time per overtake
        self.overtake_cooldown_max = overtake_cooldown  # 200 steps between bonuses
        self.collision_radius      = collision_radius   # 3.0 m
        self.collision_penalty     = collision_penalty  # -0.5/step within radius

        # ── State tracking ──────────────────────────────────────────────────────
        self._prev_action  = None
        self._prev_obs     = None
        self._track_idx    = 0
        self._prev_track_idx = 0
        self.laps_completed  = 0
        self.prev_gap        = None
        self.overtake_cooldown = 0

        # ── Pit stop state ──────────────────────────────────────────────────────
        self.pit_cooldown_remaining = 0
        self.pit_count              = 0

    # ────────────────────────────────────────────────────────────────────────────
    # OBSERVATION BUILDER
    # ────────────────────────────────────────────────────────────────────────────

    def _get_ego_obs_11d(self):
        """
        Build the standard 11D ego observation (identical to F1MultiAgentEnv).
        Also updates self._track_idx (used by _get_track_gap).
        """
        x, y = self.car.x, self.car.y
        yaw  = self.car.yaw
        v    = self.car.v

        idx, _ = closest_point(self.track, x, y)
        self._track_idx = idx

        track_angle, _ = track_tangent(self.track, idx)
        heading_error  = normalize_angle(track_angle - yaw)
        lateral_error  = signed_lateral_error(self.track, idx, x, y)

        idx_near = (idx +  5) % self.n_wpts
        angle_near, _ = track_tangent(self.track, idx_near)
        curv_near = normalize_angle(angle_near - track_angle)

        idx_mid  = (idx + 15) % self.n_wpts
        angle_mid, _ = track_tangent(self.track, idx_mid)
        curv_mid  = normalize_angle(angle_mid - track_angle)

        idx_far  = (idx + 30) % self.n_wpts
        angle_far, _ = track_tangent(self.track, idx_far)
        curv_far  = normalize_angle(angle_far - track_angle)

        progress = idx / self.n_wpts
        v_y = getattr(self.car, 'v_y', 0.0)
        r   = getattr(self.car, 'r',   0.0)

        return np.array([
            v             / 20.0,
            heading_error / np.pi,
            lateral_error / 3.0,
            np.sin(heading_error),
            np.cos(heading_error),
            curv_near     / np.pi,
            curv_mid      / np.pi,
            curv_far      / np.pi,
            progress,
            v_y           / 5.0,
            r             / 2.0,
        ], dtype=np.float32)

    def _get_track_gap(self):
        """
        Signed track gap = (opp_progress − ego_progress), wrap-corrected.
        Returns value in [-0.5, 0.5]: positive = opponent ahead, negative = ego ahead.
        Must be called after _get_ego_obs_11d (which updates self._track_idx).
        """
        ego_idx = self._track_idx
        opp_idx, _ = closest_point(self.track, self.opp_car.x, self.opp_car.y)

        raw_gap = (opp_idx - ego_idx) / self.n_wpts
        if raw_gap > 0.5:
            raw_gap -= 1.0
        elif raw_gap < -0.5:
            raw_gap += 1.0

        return float(np.clip(raw_gap, -1.0, 1.0))

    def _get_full_obs(self):
        """Build the full 14D observation. Order: [11D, tyre_life, track_gap, opp_speed_norm]."""
        obs_11d    = self._get_ego_obs_11d()
        tyre_life  = float(self.car.tyre_life)
        track_gap  = self._get_track_gap()
        opp_v_norm = float(np.clip(self.opp_car.v / 30.0, 0.0, 1.0))

        return np.concatenate([obs_11d, [tyre_life, track_gap, opp_v_norm]]).astype(np.float32)

    # ────────────────────────────────────────────────────────────────────────────
    # RESET
    # ────────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action = None
        self._prev_obs    = None
        self.laps_completed  = 0
        self.prev_gap        = None
        self.overtake_cooldown = 0
        self.pit_cooldown_remaining = 0
        self.pit_count = 0

        fixed = options.get("fixed_start", False) if options else False

        if fixed:
            # Ego: waypoint 0, track-aligned, v=5
            start = self.track[0]
            track_angle, _ = track_tangent(self.track, 0)
            self.car.reset(x=start[0], y=start[1], yaw=track_angle, v=5.0)

            # Opponent: waypoint 20, slightly ahead, track-aligned, v=5
            opp_idx = 20
            opp_start = self.track[opp_idx]
            opp_angle, _ = track_tangent(self.track, opp_idx)
            self.opp_car.reset(x=opp_start[0], y=opp_start[1], yaw=opp_angle, v=5.0)
        else:
            ego_idx = int(self.np_random.integers(0, self.n_wpts))
            ego_start = self.track[ego_idx]
            ego_yaw   = float(self.np_random.uniform(-0.17, 0.17))
            ego_v     = float(self.np_random.uniform(2.0, 6.0))
            self.car.reset(x=ego_start[0], y=ego_start[1], yaw=ego_yaw, v=ego_v)

            opp_idx   = (ego_idx + 20) % self.n_wpts
            opp_start = self.track[opp_idx]
            opp_angle, _ = track_tangent(self.track, opp_idx)
            self.opp_car.reset(x=opp_start[0], y=opp_start[1], yaw=opp_angle, v=ego_v)

        # Reset ego tyres to full grip at episode start
        self.car.reset_tyres()

        obs = self._get_full_obs()
        self._prev_track_idx = self._track_idx
        self.prev_gap = float(obs[12])   # obs[12] = track_gap

        return obs, {}

    # ────────────────────────────────────────────────────────────────────────────
    # STEP
    # ────────────────────────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1

        throttle   = float(action[0])
        steer      = float(action[1])
        pit_signal = float(action[2])

        # 1. Step ego car
        self.car.step(throttle, steer, dt=self.dt)

        # 2. Step opponent (ExpertDriver, no tyre degradation)
        opp_action = self.expert.get_action(self.opp_car)
        self.opp_car.step(opp_action[0], opp_action[1], dt=self.dt)

        # 3. Build observation
        obs = self._get_full_obs()

        # ── Lap detection for ego ───────────────────────────────────────────────
        curr_idx = self._track_idx
        if self._prev_track_idx > 150 and curr_idx < 50:
            self.laps_completed += 1
            lap_bonus = self.lap_bonus
        else:
            lap_bonus = 0.0
        self._prev_track_idx = curr_idx

        # ── Termination ─────────────────────────────────────────────────────────
        lateral_error_norm = obs[2]
        terminated = abs(lateral_error_norm) > 1.0    # > 3.0 m off centreline
        truncated  = self.step_count >= self.max_steps

        # ── Base reward ──────────────────────────────────────────────────────────
        base_reward = self.reward_fn.compute(
            obs=obs,
            prev_obs=self._prev_obs,
            action=np.array(action, dtype=np.float32),
            prev_action=self._prev_action,
            terminated=terminated,
        )

        # ── Pit stop logic ───────────────────────────────────────────────────────
        pit_reward = 0.0
        tyre_life_now = float(self.car.tyre_life)
        agent_pit = (pit_signal > 0.0 and self.pit_cooldown_remaining == 0)

        if agent_pit:
            self.pit_count += 1
            self.car.reset_tyres()
            pit_reward = self.PIT_PENALTY
            self.pit_cooldown_remaining = self.PIT_COOLDOWN_STEPS

            # Voluntary pit bonus: rewards pitting at the right time (worn tyres)
            if tyre_life_now < self.VOLUNTARY_PIT_THRESH:
                pit_reward += self.VOLUNTARY_PIT_BONUS

        if self.pit_cooldown_remaining > 0:
            self.pit_cooldown_remaining -= 1

        # ── Competitive rewards ──────────────────────────────────────────────────
        track_gap = float(obs[12])

        pos_bonus = self.position_bonus if track_gap < 0.0 else 0.0

        dist = float(np.sqrt(
            (self.car.x - self.opp_car.x) ** 2 +
            (self.car.y - self.opp_car.y) ** 2
        ))
        col_penalty = self.collision_penalty if dist < self.collision_radius else 0.0

        overtake_reward = 0.0
        if (self.prev_gap is not None
                and self.prev_gap > 0.0
                and track_gap <= 0.0
                and self.overtake_cooldown == 0):
            overtake_reward = self.overtake_bonus
            self.overtake_cooldown = self.overtake_cooldown_max

        if self.overtake_cooldown > 0:
            self.overtake_cooldown -= 1

        self.prev_gap = track_gap

        # ── Total reward ──────────────────────────────────────────────────────────
        reward = base_reward + lap_bonus + pos_bonus - col_penalty + overtake_reward + pit_reward

        # ── Bookkeeping ───────────────────────────────────────────────────────────
        self._prev_obs    = obs
        self._prev_action = np.array(action, dtype=np.float32)

        info = {
            "speed":          obs[0] * 20.0,
            "lateral_error":  obs[2] * 3.0,
            "tyre_life":      float(obs[11]),
            "track_gap":      track_gap,
            "opp_speed":      self.opp_car.v,
            "dist_to_opp":    dist,
            "crashed":        bool(terminated),
            "laps_completed": self.laps_completed,
            "pit_count":      self.pit_count,
            "overtake":       bool(overtake_reward > 0),
        }

        return obs, reward, terminated, truncated, info
