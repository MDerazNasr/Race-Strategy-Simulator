"""Track definition and geometry."""

import numpy as np

def generate_oval_track(radius = 50, points = 200):
    """Generate an oval track with the given radius and number of points."""
    #to plug into physics and RL w/out worrying about real F1 circuits yet
    angles = np.linspace(0, 2 * np.pi, points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))
def load_fastf1_track(year=2023, gp='Monaco', session_type='Q',
                      n_points=300, cache_dir=None):
    """
    Load real F1 circuit geometry via FastF1 telemetry.

    Downloads the fastest-lap position data for the specified session and
    returns a (N, 2) array of (x, y) waypoints in meters (FIA coordinate system).

    Args:
        year:         Season year (e.g. 2023).
        gp:           Grand Prix name (e.g. 'Monaco', 'Silverstone', 'Monza').
        session_type: 'Q' (qualifying, cleanest single lap), 'R' (race), 'FP1' etc.
        n_points:     Number of evenly-spaced waypoints to sample.  300 gives
                      ~11 m/pt for Monaco (3337 m circuit) — good curvature resolution.
        cache_dir:    Optional path for FastF1 cache.  Avoids re-downloading.
                      Recommended: 'fastf1_cache' (relative to project root).

    Returns:
        np.ndarray of shape (n_points, 2), dtype float32, in meters.

    Example:
        track = load_fastf1_track(2023, 'Monaco', 'Q', n_points=300,
                                   cache_dir='fastf1_cache')
        # track[0] = start/finish straight, units = meters
    """
    try:
        import fastf1
    except ImportError:
        raise ImportError(
            "fastf1 is required for real track loading. "
            "Install with: venv/bin/pip install fastf1"
        )

    if cache_dir is not None:
        fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True)

    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()

    # FastF1 position data is in tenths of a meter (decimeters).
    # Divide by 10 to convert to meters for the physics engine.
    x = pos['X'].values.astype(float) / 10.0
    y = pos['Y'].values.astype(float) / 10.0
    pts = np.column_stack((x, y))

    # Step 1: remove consecutive near-duplicate points from the raw telemetry
    # (FastF1 can have stationary frames at lap start/end or slow zones).
    keep = np.ones(len(pts), dtype=bool)
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[i - 1]) < 0.5:
            keep[i] = False
    pts = pts[keep]

    # Step 2: evenly sample n_points along the deduplicated telemetry
    idx = np.linspace(0, len(pts) - 1, n_points, dtype=int)
    sampled = pts[idx]

    # Step 3: remove any remaining consecutive near-duplicates in the final output
    # (can occur if linspace int-rounds two indices to the same value).
    keep2 = np.ones(len(sampled), dtype=bool)
    for i in range(1, len(sampled)):
        if np.linalg.norm(sampled[i] - sampled[i - 1]) < 0.5:
            keep2[i] = False
    return sampled[keep2].astype(np.float32)


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