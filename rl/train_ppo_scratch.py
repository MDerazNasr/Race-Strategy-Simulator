"""
This script trains a policy using PPO (Proximal Policy Optimization) on your custom
environment created by make_env.py

what is meant here by policy?
- A policy is a function that maps states to actions.
In reinforcement learning, the goal is to learn a policy that maximizes the expected cumulative reward.

"""

import sys
from pathlib import Path

from gymnasium.envs.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# SB3 likes a vectorized env
# DummyVecEnv is the simplest vector wrapper: runs 1 env but used the same interface
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.make_env import make_env  # creates the environement instance


def train():
    # Create the environment
    env = DummyVecEnv([make_env])
    # Create the PPO model
    # defining:
    # the policy arch
    # how PPO collects experience
    # how PPO updates the policy
    model = PPO(
        policy="MlpPolicy",  # MLP = multilayer perception (feedforward neural net), SB3 uses it default MLP arch unless you override policy_kwargs
        env=env,  # attach env you created so PPO can interacted with it
        learning_rate=3e-4,  # step size of gradient descent
        n_steps=2048,  # PPO collects rollouts of length n_steps per env.
        batch_size=256,  # PPO uses minibatches of size 256 from the collected rollout buffer
        n_epochs=10,  # After collecting the rollout (2048 samples), PPO will iterate over the data 10 times
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  #
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
    )

    # Tensorboard logging
    model.set_logger(configure("runs/ppo_scratch", ["stdout", "tensorboard"]))

    # Train
    model.learn(total_timesteps=300_000)

    # Save
    model.save("rl/ppo_scratch.zip")


if __name__ == "__main__":
    train()
