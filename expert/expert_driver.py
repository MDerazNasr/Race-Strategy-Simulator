"""Expert driver policy for generating demonstrations."""

import numpy
from env.track import closest_point

class ExpertDriver:
    def __init__(self, track, lookahead=8, max_speed=20.0, corner_factor=12.0):
        self.track = track
        self.lookahead = lookahead
        self.max_speed = max_speed
        self.corner_factor = corner_factor
    
    
        
             