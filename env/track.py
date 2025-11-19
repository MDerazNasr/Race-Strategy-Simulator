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

'''
This gives a progress ratio between 0 and 1.

Example:
	•	idx = 0 → start line → 0%
	•	idx = 50 → quarter lap → 0.25
	•	idx = 199 → almost full lap → 0.995

This is VERY crude but works for a circle.

Later we will improve this using:
	•	cumulative arc length
	•	spline-based track distance
	•	segment-based progress
	•	projection onto track centerline

Let’s Answer the “Why?” Intuitively

Why generate the track as points?

Because simulation happens in discrete time, so tracks are easier to handle as discrete points too.

Why find closest point?

Because you need a reference on the track to:
	•	compute car progress
	•	detect off-track
	•	compute reward
	•	orient the car relative to the track
	•	make racing lines
Why is progress = idx / len(track)?

It’s the simplest approximation of “how far around the lap” you are.

Better systems will measure real distance along the centerline.

But for now, this works.
'''