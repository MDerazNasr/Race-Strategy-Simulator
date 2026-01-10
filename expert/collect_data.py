"""Data collection from expert demonstrations."""
#checking if the expert even works

from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env
from expert.expert_driver import ExpertDriver
import matplotlib.pyplot as plt
import numpy as np

'''
This funct adds randomness to the experts actions

why?
- many RL algos (especially continous control) need exploration
- if the agent always takes the exact same action, it may get stuck in bad habits
- Noise helps the agent discover better actions

So this function:
- takes an action (throttle, steer)
- Adds small random noise to each component
- clamps the result to valid action bounds [-1,1]

std - standard deviation
'''
def add_action_noise(action, throttle_std=0.05, steer_std=0.05):
    #create the noise vector - get random # from Gaussian normal distribution
    #keeps noise values small
    # noise = [throttle_noise, steer_npoise]
    noise = np.array([
        np.random.normal(0, throttle_std),
        np.random.normal(0, steer_std),
    ])
    noisy_action = action + noise
    #if its < -1, set to -1, if > 1 set to 1, else leave unchanged
    #preventing noise from pushing outside this range
    noisy_action = np.clip(noisy_action, -1.0, 1.0)

    return noisy_action


def run_expert_lap(
        max_steps = 2000,
        use_noise = True,
        throttle_std = 0.05,
        steer_std = 0.05,
        record_dataset = False,
):
    '''
    runs one expert lap (episode) and plots trajectory

    max_steps: how many steps to run max
    use_noise: if True, we add small noise to expert actions
    throttle_std/steer_std: noise size
    record_dataset: if True, we store (obs, action) pairs for BC traning
    '''
    # 1 - create the env (contains track + car physics)
    env = F1Env()
    # 2 - Create the expert driver (contains track + car physics)
    expert = ExpertDriver(env.track)
    # 3 - Reset env to start episode
    obs, info = env.reset()
    xs,ys = [], []
    # if we want dataset recording, store states + actions here
    states = []
    actions = []


    for step in range(max_steps):
        #record where the car is for plotting
        xs.append(env.car.x)
        ys.append(env.car.y)

        #expert decides action based on current car state
        #expert_action is a numpy array of shape (2, )
        expert_action = expert.get_action(env.car)
        
        #add noise to make dataset more diverse
        if use_noise:
            action = add_action_noise(
                expert_action,
                throttle_std=throttle_std,
                steer_std=steer_std,
            )
        else:
            action = expert_action

        # 3 - record the (state, action) pair
        # RECORD OBS BEFORE STEPPING - obs corresponds to the state the expert used to choose the action
        if record_dataset:
            states.append(obs)
            actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        # if episode ends (off-track or max steps), stop
        if terminated or truncated:
            print(f"Episode ended at step {step}. terminated={terminated}, truncated={truncated}")
            break
    # 5 - plot track and expert path
    plt.figure(figsize=(6, 6))
    plt.plot(env.track[:, 0], env.track[:, 1], "--", alpha=0.5, label="Track")
    plt.plot(xs, ys, "r-", label="Expert Path")
    plt.axis("equal")
    plt.legend()
    plt.title("Expert Driver Trajectory")
    plt.show()

    #6 return recorded dataset
    if record_dataset:
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        print("Recorded states shape: ", states.shape)
        print("Recorded actions shape: ", actions.shape)
        return states, actions
    
    return None

if __name__ == "__main__":
    run_expert_lap()

'''
What this gives you:
	A line plot showing if the expert car can circle the track
	if the car stays on the centerline → physics + steering logic works
	If it oscillates → tune lookahead, max_speed, corner_factor

Now your environment has a working, deterministic human-designed driver.
'''