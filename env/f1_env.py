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

        #Observations: [x,y, cos(yaw), sin(yaw), v ]
        '''
        observation space, what the agent sees
        every observation here is a 7D vector:
            1.	x → car position X
            2.	y → car position Y
            3.	cos(yaw)
            4.	sin(yaw)
            5.	v → speed
            6.	dist → distance from the track centerline
            7.	progress → how far around the track you are (0 to ~1)
        
        '''
        obs_high = np.array( #max magnitude guess
            [1000, 1000, 1, 1, 100, 100, 1],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = obs_high,
            high = obs_high,
            dtype = np.float32
        )
        self.max_steps = 2000 #stop episode after 2000 timestamps to avoid infinite ep
        self.step_count = 0

    def get_obs(self):
        #pull the car state
        x, y, yaw, v = self.car.x, self.car.y, self.car.yaw, self.car.v
        #find the closest track point and the distance to it (track.py)
        idx, dist = closest_point(self.track, x, y)
        #covert index into [0,1] progress
        progress = progress_along_track(self.track, idx)
        #build observation vector
        obs = np.array(
            [
                x,
                y,
                np.cos(yaw),
                np.sin(yaw),
                v,
                dist,
                progress,
            ],
            dtype=np.float32,
        )
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
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0 #reset counter to 0

        #start near first track point, facing tangentially
        start = self.track[0]
        self.car.reset(x=start[0], y=start[1], yaw=0.0, v=0.0)  #yaw=0 → currently you’re just pointing in +x direction.

        obs = self.get_obs()
        info = self.get_info()
        return obs, info #return initial observation, and info (implemented later)
    
    #advance simulation one tick
    def step(self, action): 
        self.step_count += 1 

        throttle, steer = action
        state = self.car.step(throttle, steer, dt=self.dt) #call step() with curr throttle, steer
        obs = self.get_obs() #recompute the observation

        #reward: move forward along track, penalise distance from center & low speed
        x, y, yaw, v = state #state is [x, y, yaw, v] returned from the car.
        '''
        Currently:
            Faster car → higher reward (v * 0.1)
            Farther from track centerline → penalty (- dist * 0.05)

        So the agent is encouraged to:
            go fast
            stay near the track
        '''
        idx, dist = closest_point(self.track, x, y)
        progress = progress_along_track(self.track, idx)

        reward = v * 0.1 - dist * 0.05 #simple placeholder

        '''
        2 ways an episode could end
        1.terminated → “natural/real” ending
            Here: if the car is more than 20 units away from the track, we treat it as going off track.
            You also give a -10 penalty as a “you screwed up” signal.
        2.truncated → “artificial/time limit” ending
            If the episode just lasts too long (step_count >= max_steps)
            This is not “bad”, just a cut-off.
        '''
        terminated = False
        truncated = False

        #End episode if too far form tack or we exceed steps
        if dist > 20.0:
            terminated = True
            reward -= 10.0 #penalty for going off-track
        
        if self.step_count >= self.max_steps:
            truncated = True
        '''
            obs → what next state looks like
            reward → how good that action was
            terminated → did we end for real (off track)?
            truncated → did we time out?
            info → extra, not used for learning, but good for logging
        '''
        info = {"progress": progress, "dist": dist}
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


delete this line later

'''