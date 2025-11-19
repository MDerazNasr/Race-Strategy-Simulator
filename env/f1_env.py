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
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, dt= 0.1):
        super().__init__()
        super.dt = dt
        self.render_mode = render_mode

        #Track
        self.track = generate_oval_track()

        #Car
        self.car = Car()

        #Action: [throttle, steer_norm]
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
        )

        #Observations: [x,y, cos(yaw), sin(yaw), v ]
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