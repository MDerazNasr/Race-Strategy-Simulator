"""Car physics model and dynamics."""

import numpy as np

class Car:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.v = 0
    
    def step(self, throttle, steering, dt=0.1):
        #implement physics
        #this will becomne RL environment
        pass