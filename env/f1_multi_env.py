"""
F1MultiAgentEnv — Ego PPO vs ExpertDriver Opponent (D39).

WHY MULTI-AGENT RACING?
========================
All previous environments pit the agent against only a timer and a reward function.
Real racing is competitive: you must be FASTER THAN YOUR OPPONENT. This creates:
  - A dynamic positional reward: being ahead of the opponent is immediately valuable.
  - Overtake bonuses: the agent learns to set up and execute passes.
  - Soft collision avoidance: the agent learns not to crash into the opponent.

DESIGN DECISIONS:
=================
  1. Standalone class (does NOT inherit F1Env): keeps the code clean and avoids
     complex multi-inheritance with F1Env's pit/tyre machinery (not relevant here).
  2. 13D observation (11D standard + track_gap + opp_speed_norm): minimal extension
     so cv2 weights transfer cleanly via extend_obs_dim(model, 11, 13).
  3. 2D action [throttle, steer]: no pit signal (standard environment).
  4. Opponent: ExpertDriver at max_speed=22 m/s — same as ppo_curriculum_v2's 26.9 m/s peak,
     but not as fast, giving the PPO agent a meaningful but beatable opponent.
  5. Position bonus: +0.5/step when ahead (track_gap < 0) — continuous incentive.
  6. Overtake bonus: +200 one-time when ego transitions from behind to ahead.
     200-step cooldown prevents lapping from farming this bonus repeatedly.
  7. Collision penalty: -0.5/step within 3m — soft deterrent, not hard termination.
     Small enough not to punish the approach phase before an overtake.
  8. ent_coef=0.01 in training: prevents log_std collapse (lesson from d38).

OBSERVATION (13D):
  [0-10]  Identical to F1Env.get_obs() 11D vector (speed, heading, lateral, ...)
  [11]    track_gap ∈ [-1, 1]: (opp_progress - ego_progress), wrap-corrected.
            Positive = opponent ahead, Negative = ego ahead.
  [12]    opp_speed_norm = clip(opp.v / 30.0, 0, 1): how fast is the opponent?

TRACK GAP SIGN CONVENTION:
  track_gap = (opp_idx - ego_idx) / len(track), wrap-corrected to [-0.5, 0.5].
  Positive = opponent is ahead on current lap.
  Negative = ego is ahead.
  Wrap correction: if |raw_gap| > 0.5, the shorter arc is on the other side.

OVERTAKE DETECTION:
  If prev_gap was positive (opponent ahead) AND current gap is negative (ego ahead):
  → The ego just overtook the opponent. Award +200 bonus.
  Cooldown: 200 steps. Prevents double-counting when ego and opponent are neck-and-neck.
  Note: a full lap by the ego while the opponent is behind does NOT trigger the bonus
  (the gap goes from negative → positive → negative, first transition is wrong direction).

RESET:
  Fixed start: ego at waypoint 0 (track-aligned, v=5 m/s).
               Opponent at waypoint 20 (just ahead, track-aligned, v=5 m/s).
               → Ego starts slightly behind, must overtake.
  Random start: ego at random waypoint, random yaw ±10°, v=2–6 m/s.
                Opponent at (ego_idx + 20) % N, track-aligned, same v as ego.
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
from env.track import generate_oval_track, closest_point, progress_along_track
from utils.geometry import normalize_angle, track_tangent, signed_lateral_error
from expert.expert_driver import ExpertDriver
from rl.rewards import RacingReward


class F1MultiAgentEnv(gym.Env):
    """
    Ego PPO car vs ExpertDriver opponent on the oval track.

    Observation: 13D = [11D standard ego obs] + [track_gap] + [opp_speed_norm]
    Action:      2D  = [throttle, steer]  (no pit)
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        dt: float = 0.1,
        opp_max_speed: float = 22.0,
        position_bonus: float = 0.5,
        overtake_bonus: float = 200.0,
        overtake_cooldown: int = 200,
        collision_radius: float = 3.0,
        collision_penalty: float = 0.5,
    ):
        super().__init__()
        self.dt = dt

        # ── Track ─────────────────────────────────────────────────────────────
        self.track = generate_oval_track()
        self.n_wpts = len(self.track)

        # ── Cars ──────────────────────────────────────────────────────────────
        self.car = DynamicCar()                      # ego: PPO-controlled
        self.opp_car = DynamicCar()                  # opponent: ExpertDriver

        # ── Opponent policy ────────────────────────────────────────────────────
        # ExpertDriver at max_speed=22 m/s — faster than standard (20) but not
        # as fast as the trained cv2 PPO (26.9 m/s). Gives a beatable challenge.
        self.expert = ExpertDriver(
            self.track,
            max_speed=opp_max_speed,
            lookahead=8,
            corner_factor=12.0,
            include_pit=False,
        )

        # ── Racing reward (ego only) ───────────────────────────────────────────
        self.reward_fn = RacingReward()

        # ── Action space: [throttle, steer] ───────────────────────────────────
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # ── Observation space: 13D ─────────────────────────────────────────────
        # Dims 0-10: same bounds as F1Env (11D standard)
        # Dim 11: track_gap ∈ [-1, 1]
        # Dim 12: opp_speed_norm ∈ [0, 1]
        obs_high = np.array(
            [100.0, np.pi, 50.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0, 20.0, 5.0,
             1.0,   # track_gap
             1.0],  # opp_speed_norm
            dtype=np.float32,
        )
        obs_low = np.array(
            [0.0, -np.pi, -50.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0, -20.0, -5.0,
             -1.0,  # track_gap
             0.0],  # opp_speed_norm
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ── Episode parameters ─────────────────────────────────────────────────
        self.max_steps = 2000
        self.step_count = 0
        self.lap_bonus = 100.0

        # ── Competitive parameters ─────────────────────────────────────────────
        self.position_bonus    = position_bonus    # +0.5/step when ego ahead
        self.overtake_bonus    = overtake_bonus    # +200 one-time per overtake
        self.overtake_cooldown_max = overtake_cooldown  # 200 steps between bonuses
        self.collision_radius  = collision_radius  # 3.0 m
        self.collision_penalty = collision_penalty # -0.5/step within radius

        # ── State tracking ─────────────────────────────────────────────────────
        self._prev_action = None
        self._prev_obs = None
        self._track_idx = 0
        self._prev_track_idx = 0
        self.laps_completed = 0
        self.prev_gap = None              # previous step's track_gap
        self.overtake_cooldown = 0        # steps remaining until next bonus allowed

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ego_obs_11d(self):
        """
        Build the standard 11D ego observation (identical to F1Env.get_obs()
        without tyre_degradation).
        """
        x, y = self.car.x, self.car.y
        yaw  = self.car.yaw
        v    = self.car.v

        idx, _ = closest_point(self.track, x, y)
        self._track_idx = idx

        track_angle, _ = track_tangent(self.track, idx)
        heading_error  = normalize_angle(track_angle - yaw)
        lateral_error  = signed_lateral_error(self.track, idx, x, y)

        idx_near  = (idx +  5) % self.n_wpts
        angle_near, _ = track_tangent(self.track, idx_near)
        curv_near = normalize_angle(angle_near - track_angle)

        idx_mid   = (idx + 15) % self.n_wpts
        angle_mid, _ = track_tangent(self.track, idx_mid)
        curv_mid  = normalize_angle(angle_mid - track_angle)

        idx_far   = (idx + 30) % self.n_wpts
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
        Compute signed track gap = (opp_progress - ego_progress), wrap-corrected.

        Returns a value in [-0.5, 0.5]:
          Positive = opponent is ahead on the current lap.
          Negative = ego is ahead.

        Wrap correction: if the raw difference exceeds 0.5 (half-lap), the
        other direction is shorter, so we subtract 1.0 to get the true gap.

        NOTE: ego_idx is read from self._track_idx (cached by _get_ego_obs_11d).
        Always call _get_ego_obs_11d() before _get_track_gap() — _get_full_obs()
        enforces this order.
        """
        ego_idx = self._track_idx   # cached by _get_ego_obs_11d in _get_full_obs
        opp_idx, _ = closest_point(self.track, self.opp_car.x, self.opp_car.y)

        ego_progress = ego_idx / self.n_wpts
        opp_progress = opp_idx / self.n_wpts

        raw_gap = opp_progress - ego_progress
        # Wrap correction: shortest arc
        if raw_gap > 0.5:
            raw_gap -= 1.0
        elif raw_gap < -0.5:
            raw_gap += 1.0

        return float(np.clip(raw_gap, -1.0, 1.0))

    def _get_full_obs(self):
        """Build the full 13D observation."""
        obs_11d   = self._get_ego_obs_11d()
        track_gap = self._get_track_gap()
        opp_v_norm = float(np.clip(self.opp_car.v / 30.0, 0.0, 1.0))

        return np.concatenate([obs_11d, [track_gap, opp_v_norm]]).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action = None
        self._prev_obs = None
        self.laps_completed = 0
        self.prev_gap = None
        self.overtake_cooldown = 0

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
            # Ego: random position, random yaw ±10°, random speed 2-6 m/s
            ego_idx = int(self.np_random.integers(0, self.n_wpts))
            ego_start = self.track[ego_idx]
            ego_yaw = float(self.np_random.uniform(-0.17, 0.17))
            ego_v   = float(self.np_random.uniform(2.0, 6.0))
            self.car.reset(x=ego_start[0], y=ego_start[1], yaw=ego_yaw, v=ego_v)

            # Opponent: 20 waypoints ahead of ego, track-aligned
            opp_idx = (ego_idx + 20) % self.n_wpts
            opp_start = self.track[opp_idx]
            opp_angle, _ = track_tangent(self.track, opp_idx)
            self.opp_car.reset(x=opp_start[0], y=opp_start[1], yaw=opp_angle, v=ego_v)

        obs = self._get_full_obs()
        self._prev_track_idx = self._track_idx
        self.prev_gap = obs[11]   # initial track_gap

        return obs, {}

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1

        # 1. Apply ego action
        throttle = float(action[0])
        steer    = float(action[1])
        self.car.step(throttle, steer, dt=self.dt)

        # 2. Step opponent (ExpertDriver)
        opp_action = self.expert.get_action(self.opp_car)
        self.opp_car.step(opp_action[0], opp_action[1], dt=self.dt)

        # 3. Build observation
        obs = self._get_full_obs()

        # ── Lap detection for ego ──────────────────────────────────────────────
        curr_idx = self._track_idx   # updated by _get_ego_obs_11d inside _get_full_obs
        if self._prev_track_idx > 150 and curr_idx < 50:
            self.laps_completed += 1
            lap_bonus = self.lap_bonus
        else:
            lap_bonus = 0.0
        self._prev_track_idx = curr_idx

        # ── Termination ────────────────────────────────────────────────────────
        lateral_error_norm = obs[2]
        terminated = abs(lateral_error_norm) > 1.0     # > 3.0 m off centerline
        truncated  = self.step_count >= self.max_steps

        # ── Base reward from RacingReward ──────────────────────────────────────
        base_reward = self.reward_fn.compute(
            obs=obs,
            prev_obs=self._prev_obs,
            action=np.array(action, dtype=np.float32),
            prev_action=self._prev_action,
            terminated=terminated,
        )

        # ── Competitive rewards ────────────────────────────────────────────────
        track_gap = float(obs[11])

        # Position bonus: continuous reward for being ahead
        pos_bonus = self.position_bonus if track_gap < 0.0 else 0.0

        # Collision penalty: soft deterrent when cars are very close
        dist = float(np.sqrt(
            (self.car.x - self.opp_car.x) ** 2 +
            (self.car.y - self.opp_car.y) ** 2
        ))
        col_penalty = self.collision_penalty if dist < self.collision_radius else 0.0

        # Overtake bonus: one-time reward when ego transitions from behind to ahead
        overtake_reward = 0.0
        if (self.prev_gap is not None
                and self.prev_gap > 0.0    # was behind
                and track_gap <= 0.0       # now ahead
                and self.overtake_cooldown == 0):
            overtake_reward = self.overtake_bonus
            self.overtake_cooldown = self.overtake_cooldown_max

        if self.overtake_cooldown > 0:
            self.overtake_cooldown -= 1

        self.prev_gap = track_gap

        # ── Total reward ───────────────────────────────────────────────────────
        reward = base_reward + lap_bonus + pos_bonus - col_penalty + overtake_reward

        # ── Bookkeeping ────────────────────────────────────────────────────────
        self._prev_obs    = obs
        self._prev_action = np.array(action, dtype=np.float32)

        info = {
            "speed":          obs[0] * 20.0,
            "heading_error":  obs[1] * np.pi,
            "lateral_error":  obs[2] * 3.0,
            "v_y":            obs[9] * 5.0,
            "yaw_rate":       obs[10] * 2.0,
            "crashed":        bool(terminated),
            "laps_completed": self.laps_completed,
            "track_gap":      track_gap,
            "opp_speed":      self.opp_car.v,
            "dist_to_opp":    dist,
            "overtake":       bool(overtake_reward > 0),
        }

        return obs, reward, terminated, truncated, info
