"""Main entry point for RL-F1 driver project."""

from env.f1_env import F1Env
import numpy as np

def run_random_steps(steps=200):
    env = F1Env()
    obs, info = env.reset()
    total_reward = 0.0

    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {t}, total reward={total_reward: .2f}")
            break
    
    env.close()
    print(f"Final total rewards:", total_reward)

if __name__ == "__main__":
    run_random_steps()