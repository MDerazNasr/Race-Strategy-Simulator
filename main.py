"""Main entry point for RL-F1 driver project."""

from env.f1_env import F1Env
import numpy as np

def run_random_steps(steps=200):
    env = F1Env() #builds track, car, action space, observation space, all env internals
    obs, info = env.reset()
    total_reward = 0.0

    for t in range(steps):
        action = env.action_space.sample()
        '''
        returns something like: (always in valid range) 
        [ 0.92, -0.31 ]  → random throttle, random steering
        [ -0.8, 0.11 ]
        [ 1.0, -1.0 ]
        '''
        obs, reward, terminated, truncated, info = env.step(action)
        '''
        1.	Runs car physics using your Car.step().
        2.	Updates the car’s position.
        3.	Looks up closest point on track.
        4.	Computes distance from track.
        5.	Computes progress.
        6.	Computes reward: 
        '''
        total_reward += reward #reward system will be amended later on

        if terminated or truncated:
            print(f"Episode ended at step {t}, total reward={total_reward: .2f}")
            break
    
    env.close()
    print(f"Final total rewards:", total_reward)
    #test subject

if __name__ == "__main__":
    run_random_steps()