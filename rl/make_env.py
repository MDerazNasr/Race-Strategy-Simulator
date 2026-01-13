"""
what is SB3?
- SB3 is a reinforcement learning library that provides a
set of tools for training and evaluating reinforcement learning
algorithms.
"""

import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env


def make_env():
    """
    Creates a fresh envronment instance for SB3
    Monitor tracks episode reward/length

    why monitor? - it logs episode return and length,
    so SB3 can plot learning curves easily
    """

    env = F1Env()
    env = Monitor(env)
    return env
