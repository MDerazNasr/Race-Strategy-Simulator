"""Expert driver policy for generating demonstrations."""

import numpy as np
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from env.track import closest_point

class ExpertDriver:
    def __init__(
        self,
        track,
        lookahead=8,
        max_speed=20.0,
        corner_factor=12.0,
        include_pit=False,
        pit_threshold=0.3,
    ):
        """
        Rule-based expert F1 driver.

        Args:
            track:          List of (x, y) waypoints.
            lookahead:      Waypoints ahead to target for steering.
            max_speed:      Target speed in m/s on straights.
            corner_factor:  Speed reduction per radian of heading error.
            include_pit:    If True, add a 3rd action dimension: pit_signal.
                            Returns shape (3,) when True, (2,) when False.
                            Used for d18 pit-stop demonstrations.
            pit_threshold:  tyre_life below which the expert requests a pit.
                            Default 0.3 = pit when grip drops to 30% of new.
                            At base wear 0.0003/step, this triggers at ~2333 steps
                            of gentle driving, or ~538 steps of aggressive driving.
        """
        self.track = track
        self.lookahead = lookahead
        self.max_speed = max_speed
        self.corner_factor = corner_factor
        self.include_pit = include_pit
        self.pit_threshold = pit_threshold

    def get_action(self, car):
        """
        Compute expert action from current car state.

        Returns:
            np.ndarray of shape (2,) if include_pit=False (standard driving).
            np.ndarray of shape (3,) if include_pit=True (with pit_signal).

        Pit logic (when include_pit=True):
            pit_signal = 1.0 when car.tyre_life < pit_threshold → request pit.
            pit_signal = -1.0 otherwise → no pit.

        The expert does NOT track pit cooldown — it simply requests a pit
        whenever tyre_life is below threshold. The environment ignores the
        request during cooldown. Since fresh tyres (tyre_life=1.0) are well
        above the threshold, the expert won't request another pit right after
        one fires. This creates clean demonstrations: drive → degrade → pit
        → drive → degrade → pit.
        """
        x, y, yaw, v = car.x, car.y, car.yaw, car.v

        #Find closest point on track
        idx, dist = closest_point(self.track, x, y)
        target_idx = (idx + self.lookahead) % len(self.track)
        tx, ty = self.track[target_idx]

        #Compute target direction
        target_angle = np.arctan2(ty - y, tx - x)

        #Compute steering error
        delta = target_angle - yaw
        delta = (delta + np.pi) % (2*np.pi) - np.pi

        #Scale to steering norm [-1,1]
        max_steer = np.deg2rad(30) #!!
        steer_norm = np.clip(delta / max_steer, -1.0, 1.0)

        #Scale speed control
        target_speed = self.max_speed - self.corner_factor * abs(delta)
        target_speed = max(5.0, target_speed) #never go below 5 m/s

        #throttle controller
        throttle = np.clip((target_speed - v) / 5.0, -1.0, 1.0)

        if not self.include_pit:
            return np.array([throttle, steer_norm], dtype=np.float32)

        # ── Pit signal ─────────────────────────────────────────────────────
        # Read tyre_life from the car object (DynamicCar attribute).
        # getattr default 1.0 = assume new tyres if attribute missing.
        tyre_life = float(getattr(car, 'tyre_life', 1.0))
        pit_signal = 1.0 if tyre_life < self.pit_threshold else -1.0

        return np.array([throttle, steer_norm, pit_signal], dtype=np.float32)
    
