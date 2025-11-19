"""Track definition and geometry."""

import numpy as np

def generate_oval_track(radius = 50, points = 200):
    """Generate an oval track with the given radius and number of points."""
    #to plug into physics and RL w/out worrying about real F1 circuits yet
    angles = np.linspace(0, 2 * np.pi, points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))
#Adding helper function for Track distance
def closest_point(track, x, y):
    '''
    track shape - (N,2)
    return: 
        idx - index of closest point
        dist - euclidean distance to that point=
    '''
    diffs = track - np.array([x,y]) #track shape (N,2) - position 
    dists = np.linalg.norm(diffs, axis=1) #computes euc distance - sqrt( (xi - x)^2 + (yi - y)^2 )
    #argmin returns the index of the smallest value. (smallest is closest)
    idx = np.argmin(dists)
    return idx, dists[idx] #closest point, actual distance
#will refine later
def progress_along_track(track, idx):
    '''
    simple progress metric: idx / len(track)
    In [0,1]
    '''
    return idx/ len(track)

