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
        plot = True,
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

        #after obs update
        if record_dataset:
            # Extract lateral_error from observation (index 2)
            lateral_error = obs[2]
            
            #if far from centerline, record extra samples
            if abs(lateral_error) > 0.6:
                states.append(obs)
                actions.append(action)

        # if episode ends (off-track or max steps), stop
        if terminated or truncated:
            print(f"Episode ended at step {step}. terminated={terminated}, truncated={truncated}")
            break

    # 5 - plot track and expert path (only if requested)
    if plot:
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

def generate_dataset(
    num_episodes=50,
    max_steps=2000,
    use_noise=True,
    throttle_std=0.05,
    steer_std=0.05,
    output_path="bc/expert_data.npz",
):
    """
    Collect expert demonstrations from multiple episodes and save to .npz file.
    
    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        use_noise: Whether to add noise to expert actions
        throttle_std: Standard deviation for throttle noise
        steer_std: Standard deviation for steering noise
        output_path: Path to save the .npz file
    """
    all_states = []
    all_actions = []
    
    print(f"Collecting expert data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        states, actions = run_expert_lap(
            max_steps=max_steps,
            use_noise=use_noise,
            throttle_std=throttle_std,
            steer_std=steer_std,
            record_dataset=True,
            plot=False,  # Don't plot during batch collection
        )
        
        if states is not None and actions is not None:
            all_states.append(states)
            all_actions.append(actions)
            print(f"Episode {episode+1}/{num_episodes}: Collected {len(states)} samples")
        else:
            print(f"Episode {episode+1}/{num_episodes}: Failed to collect data")
    
    # Concatenate all episodes
    if len(all_states) > 0:
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        # Save to .npz file
        output_file = Path(project_root) / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_file, states=all_states, actions=all_actions)
        print(f"\nSaved dataset to {output_file}")
        print(f"Total samples: {len(all_states)}")
        print(f"States shape: {all_states.shape}")
        print(f"Actions shape: {all_actions.shape}")
    else:
        print("Error: No data collected!")

def generate_dataset_pit(
    num_episodes=50,
    max_steps=2000,
    output_path="bc/expert_data_pit.npz",
):
    """
    Collect expert demonstrations for the pit-stop environment (d18).

    WHAT'S DIFFERENT vs generate_dataset():
      - Environment: F1Env(tyre_degradation=True, pit_stops=True)
          12D observation (adds tyre_life at index 11)
          3D action space [throttle, steer, pit_signal]
      - Expert: ExpertDriver(include_pit=True, pit_threshold=0.3)
          Returns [throttle, steer, pit_signal] where:
            pit_signal = 1.0  when tyre_life < 0.3 (pit request)
            pit_signal = -1.0 otherwise (stay out)

    NO ACTION NOISE ON PIT SIGNAL:
      Throttle and steer get the usual Gaussian noise (std=0.05) for diversity.
      pit_signal is NOT noised — it's a binary decision and noise would
      create misleading half-hearted pit requests.  The BC network learns:
        tyre_life < 0.3 → push pit_signal toward +1.0
        tyre_life >= 0.3 → push pit_signal toward -1.0

    WHAT THE AGENT LEARNS FROM THIS DATA:
      The BC policy (12D obs → 3D actions) learns to:
        1. Follow the track (same as standard BC).
        2. Pit when tyre_life drops below 0.3 (new behavior).
      This becomes the warm-start for PPO curriculum training, which then
      refines the pitting strategy via RL reward signals.

    Args:
        num_episodes:  Number of expert episodes to collect.
        max_steps:     Maximum steps per episode.
        output_path:   Where to save the .npz dataset (relative to project root).

    Saves: bc/expert_data_pit.npz with keys 'states' (N, 12) and 'actions' (N, 3).
    """
    all_states  = []
    all_actions = []

    env = F1Env(tyre_degradation=True, pit_stops=True)
    expert = ExpertDriver(env.track, include_pit=True, pit_threshold=0.3)

    print(f"Collecting pit-stop expert data from {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        ep_states  = []
        ep_actions = []

        for step in range(max_steps):
            # Expert returns [throttle, steer, pit_signal]
            expert_action = expert.get_action(env.car)

            # Add noise to throttle and steer only — pit_signal is kept exact.
            noisy_throttle = float(np.clip(expert_action[0] + np.random.normal(0, 0.05), -1.0, 1.0))
            noisy_steer    = float(np.clip(expert_action[1] + np.random.normal(0, 0.05), -1.0, 1.0))
            pit_signal     = float(expert_action[2])   # no noise on pit decision

            action = np.array([noisy_throttle, noisy_steer, pit_signal], dtype=np.float32)

            ep_states.append(obs.copy())
            ep_actions.append(action)

            # Also record extra samples near the track edge (same diversity trick as generate_dataset).
            obs, reward, terminated, truncated, info = env.step(action)

            if abs(obs[2]) > 0.6:   # lateral_error > 0.6 normalized
                ep_states.append(obs.copy())
                ep_actions.append(action)

            if terminated or truncated:
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"{len(ep_states)} samples, "
                      f"tyre_life={info['tyre_life']:.2f}, "
                      f"pits={info['pit_count']}")
                break

        all_states.extend(ep_states)
        all_actions.extend(ep_actions)

    all_states  = np.array(all_states,  dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    output_file = Path(project_root) / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, states=all_states, actions=all_actions)

    print(f"\nSaved pit dataset to {output_file}")
    print(f"  Total samples: {len(all_states)}")
    print(f"  States shape:  {all_states.shape}   (expected: (N, 12))")
    print(f"  Actions shape: {all_actions.shape}  (expected: (N, 3))")
    return str(output_file)


if __name__ == "__main__":
    run_expert_lap()

'''
What this gives you:
	A line plot showing if the expert car can circle the track
	if the car stays on the centerline → physics + steering logic works
	If it oscillates → tune lookahead, max_speed, corner_factor

Now your environment has a working, deterministic human-designed driver.
'''