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
    Creates a fresh environment instance for SB3.
    Monitor tracks episode reward/length so SB3 can plot learning curves.

    Standard mode: episodes end after 2000 steps OR on crash.
    Used for all training up to and including d15 (ppo_curriculum_v2).
    """
    env = F1Env()
    env = Monitor(env)
    return env


def make_env_multi_lap():
    """
    Creates a multi-lap environment instance for SB3.
    Episodes ONLY end on crash — no 2000-step truncation.

    WHY A SEPARATE FACTORY?
      SB3's DummyVecEnv takes a list of callables: [make_env, make_env, ...]
      Each callable must be a zero-argument function.
      Having two named factories (make_env and make_env_multi_lap) lets the
      training scripts select the mode without any lambda gymnastics.

    Used in: rl/train_ppo_multi_lap.py (d16+)
    NOT used in: evaluate.py — evaluation keeps fixed 2000-step episodes
      so that all policies are compared on equal terms (same episode length).
    """
    env = F1Env(multi_lap=True)
    env = Monitor(env)
    return env
