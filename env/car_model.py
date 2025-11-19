"""Car physics model and dynamics."""

import numpy as np

class Car:
    # storing the car's state and its physical limits
    #creating function step to update the car's state
    def __init__(
        self,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.0,
        wheelbase=2.5,
        max_steer=np.deg2rad(30),
        max_accel=5.0,
        max_decel=-8.0,
        drag_coeff=0.1,
    ):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.drag_coeff = drag_coeff

    def rest(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def step(self, throttle, steering, dt=0.1):
        #implement physics
        #this will becomne RL environment
        pass