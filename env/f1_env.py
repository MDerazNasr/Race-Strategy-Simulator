#Gymnasium environment
'''
In RL world, everything is:
	•	an agent → decides actions
	•	an environment → reacts to actions
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .car_model import Car
from .track import generate_oval_track, closest_point, progress_along_track

class F1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30} #info used for rendering

    def __init__(self, render_mode=None, dt= 0.1):
        super().__init__()
        super.dt = dt #dt = how much simulated time passes each step(). Same as in Car
        self.render_mode = render_mode #render_mode = you'll use this later if you draw stuff

        #Track
        self.track = generate_oval_track() #you generate a simple circular-ish track (list of (x,y) points)

        #Car
        self.car = Car()

        #Action: [throttle, steer_norm]
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
        )

        #Observations: [x,y, cos(yaw), sin(yaw), v ]
        obs_high = np.array(
            [1000, 1000, 1, 1, 100, 100, 1],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = obs_high,
            high = obs_high,
            dtype = np.float32
        )
        self.max_steps = 2000
        self.step_count = 0
        
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