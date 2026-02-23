#Gymnasium environment
'''
In RL world, everything is:
	•	an agent → decides actions
	•	an environment → reacts to actions
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from env.car_model import Car, DynamicCar
from env.track import generate_oval_track, closest_point, progress_along_track
from utils.geometry import (
    normalize_angle,
    track_tangent,
    signed_lateral_error,
)
from rl.rewards import RacingReward

class F1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30} #info used for rendering

    def __init__(self, render_mode=None, dt= 0.1):
        super().__init__()
        self.dt = dt #dt = how much simulated time passes each step(). Same as in Car
        self.render_mode = render_mode #render_mode = you'll use this later if you draw stuff

        #Track
        self.track = generate_oval_track() #you generate a simple circular-ish track (list of (x,y) points)

        # ── Car ──────────────────────────────────────────────────────────
        # PART A: swapped kinematic Car → DynamicCar.
        # DynamicCar has the same .x, .y, .yaw, .v interface as Car,
        # but internally runs a 6-state dynamic model with tyre slip physics.
        # The rest of the environment code doesn't need to change because
        # DynamicCar exposes backward-compatible .yaw and .v properties.
        self.car = DynamicCar()

        #Action: [throttle, steer_norm]
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
        )

        # ── Observations: 11D vector from get_obs() ─────────────────────
        '''
        PART B + PART A: expanded observation vector.

        Original 6D obs was sufficient for a kinematic car, but the dynamic
        model adds two new state variables (v_y, r) that the agent MUST see
        to understand and control tyre slip.  We also give the agent a richer
        view of the track ahead with THREE curvature samples instead of one.

        FULL 11D OBSERVATION:
          idx  value                  what it tells the agent
          ─────────────────────────────────────────────────────────────────
          [0]  v / 20.0               how fast are we going?
          [1]  heading_error / π      are we pointed at the track?
          [2]  lateral_error / 3.0    are we on the racing line?
          [3]  sin(heading_error)     smooth angle encoding (avoids ±π wrap)
          [4]  cos(heading_error)     smooth angle encoding
          [5]  curv_near / π          track curvature  5 waypoints ahead
          [6]  curv_mid  / π          track curvature 15 waypoints ahead  ← NEW (Part B)
          [7]  curv_far  / π          track curvature 30 waypoints ahead  ← NEW (Part B)
          [8]  progress               how far around the lap are we? [0, 1] ← NEW (Part B)
          [9]  v_y / 5.0              sideways sliding speed               ← NEW (Part A)
          [10] r   / 2.0              yaw rate (spinning speed)            ← NEW (Part A)

        WHY THREE CURVATURE SAMPLES?
          One sample (5 waypoints) tells you about the NEXT corner entry.
          At high speed (19 m/s), 5 waypoints ≈ 0.5 s ahead.  For a sharp
          corner, that's barely enough to start braking.

          Adding 15-waypoint (≈1.5 s) and 30-waypoint (≈3 s) samples gives
          the agent a "track map" horizon:
            curv_near → what to do RIGHT NOW
            curv_mid  → what to prepare for in the next second
            curv_far  → can I carry speed through this section?

          This is analogous to a human driver reading the road far ahead.

        WHY v_y AND r?
          With the kinematic model, v_y = 0 by construction — the car never
          slides.  The agent had no way to see tyre slip.

          With DynamicCar, the rear can slide during hard cornering.
          Without v_y and r in the obs, the agent is flying blind: it can't
          distinguish "car gripping" from "car sliding into a spin".

          v_y > 0 = sliding right, < 0 = sliding left.
          r   > 0 = yaw rate increasing (turning left), < 0 = turning right.
          Together they let the agent detect and counter oversteer before it
          becomes a spin — the same reflex that makes a skilled F1 driver.

        WHY PROGRESS?
          Without progress, the agent has no idea where it is on the lap.
          It cannot learn lap-specific strategies ("brake early at this corner").
          Adding idx/len(track) ∈ [0, 1] gives a compressed map position.
        '''
        # Define bounds for each dimension (generous bounds — Gym clips if exceeded).
        # Normalised values will always be well within these ranges during training.
        obs_high = np.array(
            # v    h_err  lat  sin  cos  c_near c_mid c_far  prog v_y   r
            [100.0, np.pi, 50.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0, 20.0, 5.0],
            dtype=np.float32
        )
        obs_low = np.array(
            [0.0, -np.pi, -50.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0, -20.0, -5.0],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = obs_low,
            high = obs_high,
            dtype = np.float32
        )
        self.max_steps = 2000  # stop episode after 2000 steps to avoid infinite episodes
        self.step_count = 0

        # Reward function — modular so it can be swapped for curriculum learning later.
        # All reward logic lives in rl/rewards.py; the env just calls it.
        self.reward_fn = RacingReward()

        # Track the previous action so the smoothness penalty has something to compare.
        # None on the first step of each episode; RacingReward handles this gracefully.
        self._prev_action = None
        self._prev_obs = None
    '''
    in RL your agent needs an observation vector each step: a compact summary of whats going on right now

    This _get_obs() builds an observation that tells the agent:
    1. speed(v)
    2. heading error = how misaligned the car is vs the track direction
    3. lateral error = how far left/right the car is from the track center line
    4. sin/cos of heading error (helps ML handle angle wrap-around smoothly)
    5. Curvature estimate = 'is the track turning soon' 

    The policy will learn:
    - if im on the left of center and the track is bending right, steer right more
    - if heading error is large, correct steering even if im centered
    - etc.
    '''
    def get_obs(self):
        """
        Build the 11D observation vector from the current car + track state.

        DIMENSIONS:
          [0]  v / 20.0            — normalised speed
          [1]  heading_error / π   — how misaligned the car is with the track
          [2]  lateral_error / 3.0 — signed sideways distance from centreline
          [3]  sin(heading_error)  — smooth angle encoding
          [4]  cos(heading_error)  — smooth angle encoding
          [5]  curv_near / π       — curvature  5 wpts ahead  (≈ 0.5 s at 19 m/s)
          [6]  curv_mid  / π       — curvature 15 wpts ahead  (≈ 1.5 s)
          [7]  curv_far  / π       — curvature 30 wpts ahead  (≈ 3.0 s)
          [8]  progress            — lap progress ∈ [0, 1]
          [9]  v_y / 5.0           — normalised lateral (sliding) velocity
          [10] r   / 2.0           — normalised yaw rate

        CURVATURE FORMULA (same for all three lookahead distances):
          curv = normalize_angle( angle_at_lookahead - angle_at_current )
          Positive = track bending left, negative = bending right.
          Magnitude = how sharp the bend is.

        PROGRESS:
          progress = idx / len(track)
          0 = start/finish line, 1 = just before start/finish.
          A continuous signal; the agent can learn position-specific behaviour
          (e.g., "I'm at the long straight — hold max throttle").
        """
        # ── Pull current car state ─────────────────────────────────────
        x, y = self.car.x, self.car.y
        yaw  = self.car.yaw   # works for both Car (.yaw) and DynamicCar (.yaw alias)
        v    = self.car.v     # total speed — DynamicCar .v = sqrt(v_x²+v_y²)

        # ── Find closest waypoint on the track ────────────────────────
        idx, _ = closest_point(self.track, x, y)
        n_wpts = len(self.track)

        # ── Track direction and errors at current position ─────────────
        track_angle, _ = track_tangent(self.track, idx)
        heading_error  = normalize_angle(track_angle - yaw)
        lateral_error  = signed_lateral_error(self.track, idx, x, y)

        # ── PART B: Three curvature samples ────────────────────────────
        # curv_near: 5 waypoints ahead  — immediate turn entry
        idx_near  = (idx +  5) % n_wpts
        angle_near, _ = track_tangent(self.track, idx_near)
        curv_near = normalize_angle(angle_near - track_angle)

        # curv_mid: 15 waypoints ahead  — ~1.5 s planning horizon
        idx_mid   = (idx + 15) % n_wpts
        angle_mid, _ = track_tangent(self.track, idx_mid)
        curv_mid  = normalize_angle(angle_mid - track_angle)

        # curv_far: 30 waypoints ahead  — ~3.0 s planning horizon
        idx_far   = (idx + 30) % n_wpts
        angle_far, _ = track_tangent(self.track, idx_far)
        curv_far  = normalize_angle(angle_far - track_angle)

        # ── PART B: Lap progress ───────────────────────────────────────
        # Linear fraction of the lap completed: 0.0 at start, ~1.0 just before lap end.
        progress = idx / n_wpts

        # ── PART A: Dynamic state (lateral velocity + yaw rate) ────────
        # For DynamicCar these are real physics states.
        # For the old kinematic Car they are always 0.0 (backward compatible).
        v_y = getattr(self.car, 'v_y', 0.0)   # lateral sliding speed (m/s)
        r   = getattr(self.car, 'r',   0.0)   # yaw rate (rad/s)

        # ── Assemble 11D observation vector ────────────────────────────
        obs = np.array([
            v             / 20.0,   # [0]  speed
            heading_error / np.pi,  # [1]  heading error
            lateral_error / 3.0,    # [2]  lateral error
            np.sin(heading_error),  # [3]  sin(heading)
            np.cos(heading_error),  # [4]  cos(heading)
            curv_near     / np.pi,  # [5]  near curvature
            curv_mid      / np.pi,  # [6]  mid  curvature
            curv_far      / np.pi,  # [7]  far  curvature
            progress,               # [8]  lap progress
            v_y           / 5.0,    # [9]  lateral velocity
            r             / 2.0,    # [10] yaw rate
        ], dtype=np.float32)

        return obs
    '''
    Why sin(heading_error) and cos(heading_error) instead of heading_error directly?

        Angles wrap around: heading_error = +π and -π are the same direction,
        but a neural net treats them as very different numbers (large positive vs
        large negative).  sin/cos encode the angle as a point on the unit circle,
        which is continuous and smooth at all angles — no discontinuity at ±π.
    '''
    
    def get_info(self):
        return {}
    #starting new episode
    '''
    why the following ranges?
    - +/- 10 deg simulates imperfect alignemnt
    - 2-6 m/s aboids
    
    '''
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0       # reset step counter
        self._prev_action = None  # clear previous action (no smoothness penalty on step 0)
        self._prev_obs = None     # clear previous obs

        #random starting index
        idx = self.np_random.integers(0, len(self.track))
        start = self.track[idx]

        # random yaw pertubation (~+/- 10 degrees)
        yaw = self.np_random.uniform(-0.17, 0.17)

        #small random initial speed
        v = self.np_random.uniform(2.0, 6.0)

        self.car.reset(
            x = start[0],
            y = start[1],
            yaw=yaw,
            v=v,
        )

        obs = self.get_obs()
        info = {}
        return obs, info
    
    #advance simulation one tick
    '''
    1. apply the agents action (throttle + steering) to the car
    2. Advance physics by a small time dt
    3. compute observation (what the agent sees now)
    4. compute reward (how good that step was)
    5. check if episdoe should end (crash/off-track/track limit)
    6. Return (obs, reward, terminated, truncated, info)
    This is the whole RL loop
    '''
    def step(self, action):
        # Count how many steps we've taken in this episode
        self.step_count += 1

        # 1. Apply agent action to the car physics model.
        throttle, steer = action
        self.car.step(throttle, steer, dt=self.dt)

        # 2. Compute observation from the new car state.
        obs = self.get_obs()

        # 3. Determine termination before computing reward, because the
        #    terminal penalty is baked into RacingReward.compute().
        lateral_error = obs[2]  # normalized: raw / 3.0, range ≈ [-1, 1]

        # Terminate if car goes more than 3.0 m off centerline.
        # lateral_error is normalized by 3.0, so threshold = 1.0 normalized.
        # (Previous bug: threshold was 3.0 on the normalized value = 9m raw,
        #  which almost never fired and broke the terminal learning signal.)
        terminated = abs(lateral_error) > 1.0   # 1.0 normalized = 3.0 m raw
        truncated  = self.step_count >= self.max_steps

        # 4. Compute shaped reward via the modular RacingReward class.
        #    All reward math and hyperparameters live in rl/rewards.py.
        reward = self.reward_fn.compute(
            obs=obs,
            prev_obs=self._prev_obs,
            action=np.array(action, dtype=np.float32),
            prev_action=self._prev_action,
            terminated=terminated,
        )

        # 5. Store current obs/action for next step's smoothness penalty.
        self._prev_obs    = obs
        self._prev_action = np.array(action, dtype=np.float32)

        # 6. Extra debug info for logging / evaluation.
        info = {
            "speed":         obs[0] * 20.0,   # un-normalize back to m/s for readability
            "heading_error": obs[1] * np.pi,  # un-normalize back to radians
            "lateral_error": obs[2] * 3.0,    # un-normalize back to meters
            # PART A: expose dynamic states for TensorBoard and evaluation
            "v_y":           obs[9]  * 5.0,   # lateral sliding speed (m/s)
            "yaw_rate":      obs[10] * 2.0,   # yaw rate (rad/s)
        }
        return obs, reward, terminated, truncated, info

    #skip full rendering for now
    def render(self):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            self.ax.plot(self.track[:,0], self.track[:,1], '--', alpha=0.5)

            (self.car_point,) = self.ax.plot([], [], 'ro')
            self.ax.set_aspect('equal')
        self.car_point.set_data(self.car.x, self.car.y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clone(self):
        pass
'''
Gym gives a standard interface:
	•	reset() → start a new episode → returns initial observation
	•	step(action) → take one step → returns:
	•	obs (what agent sees next)
	•	reward (how good that step was)
	•	terminated (did we reach a “real” end? crash, finish lap, etc.)
•	truncated (did we stop because of a time limit, not a real ending?)
	•	info (extra debug info)

Your F1Env is wrapping:
	•	the Car physics model
	•	the Track
	•	the reward logic

into something an RL algorithm can use.

'''
