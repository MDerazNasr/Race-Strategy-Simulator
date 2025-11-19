import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .car_model import Car
from .track import generate_oval_track, closest_point, progress_along_track