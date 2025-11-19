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
        self.yaw = yaw #heading from the telemetry
        self.v = v #car instantaneous speed

        #this is not a true F1 physics model
        #using kinematic bicycle model -- 2 wheels front and rear
        self.wheelbase = wheelbase #distance between front and rear axle
        self.max_steer = max_steer #max steering angle
        self.max_accel = max_accel #max engine acceleration
        self.max_decel = max_decel #max braking force
        self.drag_coeff = drag_coeff #aerodynamic drag

    #resets car back to its initial state
    def rest(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
    #Where is the car after applying throttle + steer for dt seconds?
    def step(self, throttle, steering, dt=0.1):
        #implement physics
        #this will becomne RL environment
        '''
        throttle in [-1, 1] (negative = brake)
        steer_norm in [-1,1] -> scaled to [-max_steer, max_steer]
        dt: timestep (sec)
        '''
        # 1 - clip inputs
        '''
        throttle = -1 to 1
            1 = full gas
            0 = coasting
            -1 = full brake
        steer_norm = -1 to 1
            1 = full left
            -1 = full right
        '''
        #preventing unrealistic inputs
        throttle = float(np.clip(throttle, -1.0, 1.0))
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))

        # 2 - map to physical quantities

        #steering input (-1 to 1) becomes angle in radians
        # -30° ← steering → +30°
        steering = steer_norm * self.max_steer
        if throttle >= 0:
            accel = throttle * self.max_accel # +ve throttle pushes car
        else:
            accel = throttle * (-self.max_decel) # -ve throttle brakes/slows down car

        # 3 - simple longitudinal dynamics (v-dot = accel - drag)
        '''
        v_new = v_old + (accel - drag) * dt
        
        At high speed, drag grows and reduces acceleration
	    Braking makes v go down
	    v cannot go negative (no reverse)
        '''
        dv = accel - self.drag_coeff * self.v
        self.v += dv * dt
        self.v = max(self.v, 0.0)
