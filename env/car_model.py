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
            If you steer, the front "wheel" points left/right.
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
        
        # yaw_new = yaw_old + (how fast we turn) * (how long we're turning)
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
        every time you call step(), you simulate 0.1 seconds of the car's motion.

        If you call it 10 times → that's 1 second of driving.

        If you call it 100 times → that's 1 lap worth of micro-updates.

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

        Yaw rate = how fast the car's direction changes
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
            •	Says: "if steering is basically zero, don't bother turning"

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


# =============================================================================
# PART A: DYNAMIC BICYCLE MODEL
# =============================================================================
class DynamicCar:
    """
    Dynamic bicycle model — replaces the kinematic Car with real tyre physics.

    WHAT'S NEW VS THE KINEMATIC Car CLASS?
    ========================================
    The kinematic Car assumed:
      - The car always moves exactly where it points (no sideways sliding)
      - No tyre slip, no oversteer, no understeer
      - Unrealistically grippy — perfect traction at all speeds

    DynamicCar adds:
      - Lateral velocity (v_y): the car CAN slide sideways when pushed hard
      - Yaw rate (r): how fast the heading is rotating (rad/s)
      - Tyre slip angles (α): the difference between where a tyre POINTS
        and where the car is actually MOVING at that axle
      - Pacejka tyre forces: a realistic curve where lateral grip peaks
        at a moderate slip angle and then DROPS as the tyre saturates

    STATE VECTOR: [x, y, psi, v_x, v_y, r]   (6 states vs 4 before)
      x, y   = world-frame position (m)
      psi    = heading angle (rad)        — same as yaw in the old model
      v_x    = longitudinal velocity (forward speed, m/s)
      v_y    = lateral velocity (sideways speed, m/s)
      r      = yaw rate (rad/s)           — how fast the car is spinning

    WHY DOES THIS MATTER FOR RACING?
    ==================================
    Real F1 physics at 200+ km/h:
      - Lateral G-forces are enormous (up to 5 G in corners)
      - Tyres have limited grip — once the slip angle exceeds the peak,
        lateral force DROPS (tyre is sliding, not gripping)
      - This creates classic oversteer/understeer behaviour

    For RL training:
      - v_y and r in the observation tell the agent "the rear is sliding"
      - The agent can learn to counter-steer before a spin develops
      - Policies trained on dynamic physics generalise better to real cars
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
        v_x: float = 0.0,
        v_y: float = 0.0,
        r: float = 0.0,
        # ── Vehicle parameters (roughly F1-scale) ─────────────────────
        m: float = 750.0,           # total mass (kg)         — F1 car ~798 kg w/ driver
        I_z: float = 1200.0,        # yaw moment of inertia (kg·m²) — resistance to spinning
        a: float = 1.3,             # front axle to centre-of-mass distance (m)
        b: float = 1.2,             # rear axle  to centre-of-mass distance (m)
        C_f: float = 80_000.0,      # front cornering stiffness (N/rad)
        C_r: float = 80_000.0,      # rear  cornering stiffness (N/rad)
        mu: float = 1.5,            # peak tyre friction coefficient (F1 slicks ≈ 1.5-2.0)
        max_steer: float = np.deg2rad(20),  # max physical steering angle (rad) — F1 ≈ 15-20°
        max_accel: float = 15.0,    # max longitudinal acceleration (m/s²) — F1 ≈ 15 m/s²
        max_decel: float = -20.0,   # max braking deceleration  (m/s²) — F1 ≈ -20 m/s²
        drag_coeff: float = 0.02,   # aero drag coefficient (lower than kinematic — less damping)
    ):
        # ── Initial state ──────────────────────────────────────────────
        self.x   = x
        self.y   = y
        self.psi = psi    # heading angle (rad) — aliased as .yaw for backward compatibility
        self.v_x = v_x    # forward speed  (m/s)
        self.v_y = v_y    # sideways speed (m/s)
        self.r   = r      # yaw rate       (rad/s)

        # ── Vehicle geometry and dynamics parameters ───────────────────
        self.m         = m
        self.I_z       = I_z
        self.a         = a           # front axle distance from CoM
        self.b         = b           # rear  axle distance from CoM
        self.C_f       = C_f
        self.C_r       = C_r
        self.mu        = mu
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.drag_coeff = drag_coeff

    # ── Backward-compatible aliases ────────────────────────────────────
    @property
    def yaw(self) -> float:
        """
        Alias so any existing code that reads .yaw still works.
        DynamicCar stores heading as .psi (physics convention),
        but the old Car class used .yaw.  Both refer to the same angle.
        """
        return self.psi

    @yaw.setter
    def yaw(self, value: float):
        self.psi = value

    @property
    def v(self) -> float:
        """
        Total speed magnitude = sqrt(v_x² + v_y²).

        The old Car had a single scalar .v.
        DynamicCar has two components, but .v lets old code still get a
        meaningful speed value without any changes.
        """
        return float(np.sqrt(self.v_x ** 2 + self.v_y ** 2))

    # ── Reset ──────────────────────────────────────────────────────────
    def reset(self, x=0.0, y=0.0, yaw=None, psi=None, v=None, v_x=None):
        """
        Reset the dynamic car to a fresh initial state.

        DUAL-NAME PARAMETERS:
          Accepts both the old Car API names (yaw, v) and new names (psi, v_x)
          so the environment's reset() can stay unchanged.

          Priority: new-style name wins if both are supplied.
            yaw / psi  → sets heading (either name works)
            v   / v_x  → sets forward speed (either name works)

        After reset:
          v_y = 0.0  (car is not sliding sideways at the start)
          r   = 0.0  (car is not spinning at the start)
        """
        self.x   = x
        self.y   = y
        self.psi = psi if psi is not None else (yaw if yaw is not None else 0.0)
        self.v_x = v_x if v_x is not None else (v   if v   is not None else 0.0)
        self.v_y = 0.0   # always zero lateral velocity at start of episode
        self.r   = 0.0   # always zero yaw rate at start of episode

    # ── Pacejka "Magic Formula" tyre model ────────────────────────────
    def _pacejka_lateral_force(self, alpha: float, F_z: float, C_tyre: float) -> float:
        """
        Compute the lateral (cornering) tyre force using the Pacejka Magic Formula.

        THE PACEJKA MAGIC FORMULA (simplified):
        =========================================
          F_y = D * sin( C * arctan( B * alpha ) )

        Where:
          alpha (α) = tyre slip angle in radians
                      — how much the tyre is sliding sideways relative to its motion
          D = peak force = mu * F_z
              (scales with tyre load: heavier car → more grip, proportionally)
          C = shape factor = 1.3  (controls how sharp the peak is)
          B = stiffness factor = C_tyre / (C * D)
              (chosen so the small-angle slope equals the linear cornering stiffness)

        PHYSICAL BEHAVIOUR:
          small α  → F_y ≈ C_tyre * alpha   (linear grip phase — tyre is working)
          peak α   → F_y = D = mu * F_z     (maximum grip)
          large α  → F_y drops below peak   (tyre saturates — sliding, not gripping)

        This models the classic "tyre curve":

          Lateral
          Force  ^
                 |      *
                 |    *   *
                 |  *       *
                 | *          ****
                 +--------------------> slip angle
                   grip  peak  sliding

        Args:
            alpha:   Tyre slip angle (rad)
            F_z:     Normal (vertical) tyre load (N)
            C_tyre:  Cornering stiffness for this axle (N/rad)  — C_f for front, C_r for rear

        Returns:
            Lateral force in Newtons, same sign as alpha.
        """
        D = self.mu * F_z                           # peak force (N)
        C = 1.3                                     # shape factor
        B = C_tyre / (C * D + 1e-6)                 # stiffness factor (+1e-6 avoids ÷0 when D≈0)
        return D * np.sin(C * np.arctan(B * alpha))

    # ── One-step dynamics integration ─────────────────────────────────
    def step(self, throttle: float, steer_norm: float, dt: float = 0.1):
        """
        Advance the dynamic bicycle model by one timestep dt.

        EQUATIONS OF MOTION (body-frame Newton's laws):
        =================================================

        The car is treated as a rigid body.  Forces act at two points:
          - Front tyre contact patch: produces F_yf (lateral) and contributes to F_drive
          - Rear  tyre contact patch: produces F_yr (lateral)

        1. LONGITUDINAL (along the car's forward axis):
             m * v_x_dot = F_drive - F_drag - F_yf * sin(delta)
           Simplified (sin(delta) ≈ 0 for small steering angles):
             v_x_dot = (F_drive - F_drag) / m

        2. LATERAL (perpendicular to the car, in body frame):
             m * (v_y_dot + v_x * r) = F_yf * cos(delta) + F_yr
           The "v_x * r" term is the Coriolis / centripetal acceleration.
           Rearranged:
             v_y_dot = (F_yf + F_yr) / m - v_x * r

        3. YAW (rotational, about the vertical axis through CoM):
             I_z * r_dot = a * F_yf * cos(delta) - b * F_yr
           Front force creates a yaw moment (distance a from CoM).
           Rear  force opposes it (distance b from CoM).

        SLIP ANGLES (how much each tyre slides):
        ==========================================
          Front:  alpha_f = delta - arctan( (v_y + a*r) / v_x )
                  = steering angle minus the angle the front of the car moves
          Rear:   alpha_r =         -arctan( (v_y - b*r) / v_x )
                  = minus the angle the rear of the car moves (no steering at rear)

        Args:
            throttle:   float in [-1, 1]   (-1 = full brake, +1 = full throttle)
            steer_norm: float in [-1, 1]   (-1 = full right, +1 = full left)
            dt:         timestep in seconds (default 0.1 s)

        Returns:
            np.ndarray shape (6,): [x, y, psi, v_x, v_y, r]
        """
        # ── 1. Clip inputs to physical limits ─────────────────────────
        throttle   = float(np.clip(throttle,   -1.0, 1.0))
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))

        # ── 2. Map normalised inputs to physical quantities ────────────
        delta = steer_norm * self.max_steer   # steering angle in radians

        if throttle >= 0:
            F_drive = throttle * self.m * self.max_accel          # traction force (N)
        else:
            F_drive = throttle * self.m * abs(self.max_decel)     # braking force  (N, negative)

        # ── 3. Tyre normal loads (static weight distribution) ─────────
        # Weight is distributed between front and rear based on CoM position.
        # Front axle supports b/(a+b) of the weight (lever rule).
        # Rear  axle supports a/(a+b) of the weight.
        # NOTE: A real car adds load transfer under braking/cornering — skip for now.
        g = 9.81
        F_zf = self.m * g * self.b / (self.a + self.b)   # front normal load (N)
        F_zr = self.m * g * self.a / (self.a + self.b)   # rear  normal load (N)

        # ── 4. Tyre slip angles ────────────────────────────────────────
        # Use max(v_x, 0.5) to avoid division by zero when the car is nearly stopped.
        # At very low speeds, slip angle physics break down anyway.
        v_x_safe = max(self.v_x, 0.5)

        # Front slip: steering angle minus the actual velocity angle at the front axle
        alpha_f = delta - np.arctan2(self.v_y + self.a * self.r, v_x_safe)

        # Rear slip: there is no steering at the rear, only body/yaw motion
        alpha_r = -np.arctan2(self.v_y - self.b * self.r, v_x_safe)

        # ── 5. Pacejka lateral tyre forces ─────────────────────────────
        F_yf = self._pacejka_lateral_force(alpha_f, F_zf, self.C_f)
        F_yr = self._pacejka_lateral_force(alpha_r, F_zr, self.C_r)

        # ── 6. Equations of motion → accelerations ────────────────────
        F_drag  = self.drag_coeff * self.m * self.v_x ** 2   # aerodynamic drag (N)

        v_x_dot = (F_drive - F_drag - F_yf * np.sin(delta)) / self.m
        v_y_dot = (F_yf * np.cos(delta) + F_yr) / self.m - self.v_x * self.r
        r_dot   = (self.a * F_yf * np.cos(delta) - self.b * F_yr) / self.I_z

        # ── 7. Euler integration ───────────────────────────────────────
        self.v_x += v_x_dot * dt
        self.v_y += v_y_dot * dt
        self.r   += r_dot   * dt

        # ── 8. Clamp states to physical limits ────────────────────────
        self.v_x = max(self.v_x, 0.0)          # no reverse gear
        self.v_x = min(self.v_x, 80.0)         # cap at 80 m/s ≈ 288 km/h
        self.v_y = np.clip(self.v_y, -20.0, 20.0)   # prevent numerical explosion
        self.r   = np.clip(self.r,   -5.0,   5.0)   # ≈ ±286 deg/s max yaw rate

        # ── 9. Update heading and world-frame position ─────────────────
        # Heading changes at the yaw rate
        self.psi += self.r * dt

        # Position update accounts for both v_x (forward) and v_y (sideways):
        #   x_dot = v_x * cos(psi) - v_y * sin(psi)
        #   y_dot = v_x * sin(psi) + v_y * cos(psi)
        # This is the body-to-world rotation matrix applied to [v_x, v_y].
        self.x += (self.v_x * np.cos(self.psi) - self.v_y * np.sin(self.psi)) * dt
        self.y += (self.v_x * np.sin(self.psi) + self.v_y * np.cos(self.psi)) * dt

        return np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r],
                        dtype=np.float32)