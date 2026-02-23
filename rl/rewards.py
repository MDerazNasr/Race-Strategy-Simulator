"""Reward function definitions for RL."""
import numpy as np

class RacingReward:
    def __init__(
        self,
        progress_weight: float = 1.0,
        speed_weight: float = 0.1,
        lateral_weight: float = 0.5,
        heading_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        terminal_penalty: float = 20.0,
    ):
