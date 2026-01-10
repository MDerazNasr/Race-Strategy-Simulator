"""Geometric utilities for track and car calculations."""

import numpy as np
'''
Goal is to understand direction of track (tangent)
2. compute a left/right error relative to the track (lateral error)
3. Keep angles nicely wrapped

'''

#angle in radians
'''
angles can become huge or wrap around
- 0 rad = facing right
- pi rad = facing left
- 2pi radians = same as 0 again

if you comoute angle differences, you can accidentally get weird jumps like
- desired - current = 6.20 rad (almost 2pi) when the real diff is -.08 rad
so this function forces any angle into a clean standard range [-pi,+pi]
'''
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

#track is a 2d list of coordinates
'''
at waypoint idx we look at:
- the curr point (p_curr)
- next point (p_next) (wrapping aorund the end)
Then we compute the tangent vector:
- tangent = the direction from current point to the next point

Then we convert that direction into an angle using arctan2 which gives:
- angle - heading angle of track at idx
- tangent - the direction of the vector


'''
def track_tangent(track, idx):
    p_curr = track[idx]
    # % len(track) = wrap around
    # if idx is at the end, modulo stops out of bounds error - takes it back to 0
    p_next = track[(idx + 1) % len(track)]
    tangent = p_next - p_curr
    # tracks local direction
    # returns angle of vector in radians
    '''
   why arctan2 not arctan
   - arctan2 correctly handles quadrants (left/right/up/down) 
   - avoids division by zero issues when x=0
    '''
    #angle is the track heading angle at this segment
    angle = np.arctan2(tangent[1], tangent[0])
    return angle, tangent

'''
this computes how far your car position (x,y) is from the track at waypoint idx
measured perpendicular to the track direction.
- positive error means youre on one side of the track
- negative means the other side
'''
def signed_lateral_error(track, idx, x, y):
    p = track[idx]
    # track tangent returns angle, tangent
    _, tangent = track_tangent(track, idx)

    #normal vector (perpendicular to tangent)
    normal = np.array([-tangent[1], tangent[0]])
    '''
    np.linalg.norm(normal) = length (magnitude) of the vector
    For [a, b] it’s sqrt(a^2 + b^2)

    normal /= ... means “divide the vector by its length”

    This turns normal into a unit vector (length = 1).

    Why do this?
    Because then the dot product gives you a distance in meters/units, not scaled by segment length.
    '''
    normal /= np.linalg.norm(normal)
    # this is the vector from the track point to the car
    #p[0] = px
    # p[1] = py
    # x - px, y - py = displacement
    # So error_vec = “where am I relative to the track point?”
    error_vec = np.array([x - p[0], y - p[1]])
    '''
    find dot product to find how much of the displacement is sideways from the track
    + - car is on the side normal is pointing towards
    - - car is on the opposite side
    '''
    return np.dot(error_vec, normal)