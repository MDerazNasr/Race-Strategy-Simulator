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
    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
    #Where is the car after applying throttle + steer for dt seconds?
    def step(self, throttle, steer_norm, dt=0.1):
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
        self.v += dv * dt #dt is timestamp, d (delta/change)
        self.v = max(self.v, 0.0) #if the braking + drag tries to make v -> 0 we just stop

        #4 kinematic bicycle model
        '''
            If you steer, the front “wheel” points left/right.
            This creates a turning yaw rate.
            Faster speed → more turning
            Bigger steering angle → more turning
            Bigger wheelbase → less turning

            So:
            small cars turn sharply
            F1 cars (long wheelbase) turn less sharply        
        '''
        if abs(steering) > 1e-4:
            beta = 0.0
            yaw_rate = self.v / self.wheelbase * np.tan(steering)
        else:
            yaw_rate = 0.0
        
        # yaw_new = yaw_old + (how fast we turn) * (how long we’re turning)
        self.yaw += yaw_rate * dt

        #5 - update position
        '''
        cos = how much movement along x direction
	    sin = how much along y direction
        '''
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt

        return np.array([self.x, self.y, self.yaw, self.v], dtype=np.float32) #[ x, y, yaw_angle, speed ]

        '''
        PURPOSE OF DT: 
        every time you call step(), you simulate 0.1 seconds of the car’s motion.

        If you call it 10 times → that’s 1 second of driving.

        If you call it 100 times → that’s 1 lap worth of micro-updates.

        Think of it like a game loop:
        The smaller the dt → the smoother the simulation.

        In Formula 1 simulators, dt is often 1ms to 10ms.
        For early versions of your project, 0.1s is fine.

        drag — air resistance
        real drag is - drag ∝ v²
        for simplicity we are using
        drag ∝ v (This prevents the car from accelerating forever.)


        dv = accel - drag
        means:
        •	Engine gives positive acceleration
        •	Aerodynamics subtracts some
        •	Net result = dv

        Yaw = direction the car is facing
        Yaw angle is measured in radians.
            •	yaw = 0 → pointing right
            •	yaw = π/2 → pointing up
            •	yaw = -π/2 → pointing down

        Yaw rate = how fast the car’s direction changes
        	•	tan(steering) → bigger steering angle → sharper turning

            
        beta - Slip angle = the difference between where the tires point vs where the car moves.
        Real racing physics:
            •	On a corner, the car does not move in the exact direction a wheel is pointing.
            •	Tyres deform → you get a slip angle.
            •	Slip angle gives you understeer/oversteer dynamics.

        For now, you ignore this → simple model.

        Beta will be used later when you add tyre physics.

        what is --> if abs(steering) > 1e-4:
            Because tan(0) = 0
        But tan(very small number) ≈ small number.

        However, floating point numbers are messy.
        Steering can be super close to zero but not exactly zero.

        Example: steering = 0.000000000145
        If you compute tan(steering) you get some tiny noise.

        This line:
            •	Prevents unnecessary math
            •	Avoids division weirdness
            •	Says: “if steering is basically zero, don’t bother turning”

        So yaw_rate becomes 0.


        Search up why cos computes x direction and sin computes y direction
        
        If you move forward with speed v and direction θ:
            •	The horizontal movement = v * cos(θ)
            •	The vertical movement = v * sin(θ)

        cos = adjacent / hypotenuse
        sin = opposite / hypotenuse
        ex:
        cos(0) = 1
        sin(0) = 0
        => car moves purely to the right
        '''