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
from env.car_model import Car
from env.track import generate_oval_track, closest_point, progress_along_track
from utils.geometry import (
    normalize_angle,
    track_tangent,
    signed_lateral_error,
)

class F1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30} #info used for rendering

    def __init__(self, render_mode=None, dt= 0.1):
        super().__init__()
        self.dt = dt #dt = how much simulated time passes each step(). Same as in Car
        self.render_mode = render_mode #render_mode = you'll use this later if you draw stuff

        #Track
        self.track = generate_oval_track() #you generate a simple circular-ish track (list of (x,y) points)

        #Car
        self.car = Car() #create a car using the model you wrote earlier

        #Action: [throttle, steer_norm]
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
        )

        #Observations: 6D vector from get_obs()
        '''
        observation space, what the agent sees
        every observation here is a 6D vector:
            1.	v → speed
            2.	heading_error → how misaligned the car is vs the track direction
            3.	lateral_error → signed distance from centerline
            4.	sin(heading_error) → for smooth angle representation
            5.	cos(heading_error) → for smooth angle representation
            6.	curvature → estimate of track curvature ahead
        
        '''
        # Define bounds for each dimension: [v, heading_error, lateral_error, sin, cos, curvature]
        obs_high = np.array(
            [100.0, np.pi, 50.0, 1.0, 1.0, np.pi],
            dtype=np.float32
        )
        obs_low = np.array(
            [0.0, -np.pi, -50.0, -1.0, -1.0, -np.pi],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = obs_low,
            high = obs_high,
            dtype = np.float32
        )
        self.max_steps = 2000 #stop episode after 2000 timestamps to avoid infinite ep
        self.step_count = 0
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
       #Pull current car state from the environment
       x,y, yaw, v = self.car.x, self.car.y, self.car.yaw, self.car.v

       #Find the closest waypoint on the track to the car position
       #closest_point likely returns (index, ___)
       idx, _ = closest_point(self.track, x,y)

       #Get direction of the track aty that closest point
       #track_tangent returns (track_angle, tangent_vector)
       track_angle, _ = track_tangent(self.track, idx)

       #Heading error: how misaligned the car is compared to the track direction
       #normalise to avoid wrap-around issues at +/- pi
       heading_error = normalize_angle(track_angle - yaw)

       #Lateral error - signed sideways distance from the track
       #Positive/Negative indicates left/right relative to track direction
       lateral_error = signed_lateral_error(self.track, idx, x,y)

       #Simple curvature estimate using a lookahead point
       # Look 5 waypoints ahead (wrap around the track if needed)
       idx2 = (idx + 5) % len(self.track)

       #Track direction at the lookahead point
       angle2, _ = track_tangent(self.track, idx2)

       #Change in track direction = how much the track will turn soon
       curvature = normalize_angle(angle2 - track_angle)

       #Build observation vector for the RL policy
       #include sin/cos of heading_error to give a smooth angle rep.
       obs = np.array([ #divisions are to normalize all numbers
           v / 20.0,
           heading_error / np.pi,
           lateral_error / 3.0,
           np.sin(heading_error),
           np.cos(heading_error),
           curvature / np.pi,
       ], dtype=np.float32)
       
       return obs
    '''
    Why cos(yaw) and sin(yaw) instead of yaw directly?

        because angles wrap around
        yaw = π and yaw = -π are basically the same direction,
        but a neural net might treat them as very different numbers.

        Using (cos, sin) gives a smooth, continuous representation.
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
        self.step_count = 0 #reset counter to 0

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
        # 1- apply agent action to the car
        # action is expected to be (throttle, steer)
        throttle, steer = action
        self.car.step(throttle, steer, dt=self.dt) #call step() with curr throttle, steer
       
        # 2 - compute new observation after ther movement
        obs = self.get_obs() #recompute the observation

        # 3 - extract key values from the observation vector
        v = obs[0] #speed
        heading_error = obs[1] # car heading vs track direction
        lateral_error = obs[2] # signed distance from centerline

        '''
        4 - reward
        - encourage forward progress along track direction
        - penalise being far from center
        - penalise pointing away from track direction
        '''
        reward = (
            v * np.cos(heading_error)
            - 0.5 * abs(lateral_error)
            - 0.1 * abs(heading_error)
        )
        # 5 - episode termination flags (Gym style)
        terminated = False # ended due to failure (off track)
        truncated = False # Ended due to time limit

        #If too far from the track centerline, and episode and punish strongly
        if abs(lateral_error) > 3.0:
            terminated = True
            reward -= 20.0
        #if we hit max steps, end due to time limit
        if self.step_count >= self.max_steps:
            truncated = True

        # Extra debug info (not usually used by the policy directly)
        info = {
            "speed": v,
            "heading_error": heading_error,
            "lateral_error": lateral_error,
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
