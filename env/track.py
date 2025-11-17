"""Track definition and geometry."""

import numpy as np

def generate_oval_track(radius = 50, points = 200):
    """Generate an oval track with the given radius and number of points."""
    #to plug into physics and RL w/out worrying about real F1 circuits yet
    angles = np.linspace(0, 2 * np.pi, points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))
