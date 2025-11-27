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
    def __init__(self, track, lookahead=8, max_speed=20.0, corner_factor=12.0):
        self.track = track
        self.lookahead = lookahead
        self.max_speed = max_speed
        self.corner_factor = corner_factor
    
    def get_action(self, car):
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

        return np.array([throttle, steer_norm], dtype=np.float32)
    
