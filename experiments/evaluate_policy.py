import numpy as np
import torch
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.f1_env import F1Env
from expert.expert_driver import ExpertDriver
from bc.train_bc import BCPolicy


def rollout_episode(env, policy_fn, max_steps=2000):
    '''
    runs on one episode using policy(fn) -> action
    , collects metrics and Returns collected metrics
    '''
    #must do this for every new episode
    #info is for debugging
    obs, info = env.reset() #obs = observations (after action)

    total_reward = 0.0
    lateral_errors = []
    speeds = []

    for t in range(max_steps):
        action = policy_fn(env, obs)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        lateral_errors.append(abs(obs[2])) #Normalized lateral error
        #speed is obs[0]
        speeds.append(obs[0]) #normalised speed

        if terminated or truncated:
            break
    '''
    if terminated = true:
    - episode ended due to failure, lap not completed

    if terminated = false:
    - episode ended due to truncationm or max steps, considered a succesful lap
    
    '''
    completed_lap = not terminated
    '''
    mean_lateral_error: measures stability
    completed_lap: binary success metric
    total_reward: overall quality
    steps: longevity
    mean_speed: aggression vs caution
    '''
    return {
        "steps": t + 1, #because its zero based
        "total_reward": total_reward,
        "mean_lateral_error": np.mean(lateral_errors),
        "mean_speed": np.mean(speeds),
        "completed_lap": completed_lap,
    }

def expert_policy(env, obs):
        return env.expert.get_action(env.car)
'''
we have a trained behaviour cloning neural network policy:
- input: state/ obsveration (size 6)
- output: action (throttle, steer) (size 2)

But:
- your rollout code expects a function:
    policy_fn(envm obs) -> action
- model expects a PyTorch tensor not np array
need to handle:
- device placement (cpu v gpu)
- batch dimensions
- disabling gradients during inference

load_bc..() - loads trained NN weights, puts model in inference mode
bc_policy() - wraps the pytorch model into a callable policy funct, converts np -> torch -> np

'''
def load_bc_policy(model_path, device):
    #Creating a new instance of your policy neural network class
    policy = BCPolicy(state_dim=6, action_dim=2).to(device)
    #imp loading step, has trained weights from BC
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval() #switches to eval mode, makes inference determinstic and stable
    return policy
#read about nested functions
def bc_policy_fn(policy, device):
    def _policy(env, obs):
        state = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action = policy(state)
        # Remove batch dimension: [1, 2] -> [2]
        return action.cpu().numpy()[0]
    return _policy

def evaluate(policy_type="expert", episodes=20):
    env = F1Env()
    results = []
    
    if policy_type == "expert":
       env.expert = ExpertDriver(env.track)
       policy_fn = lambda env, obs: env.expert.get_action(env.car)

    elif policy_type == "bc":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bc_policy = load_bc_policy("bc/bc_policy.pt", device)
        policy_fn = bc_policy_fn(bc_policy, device)
    else:
        raise ValueError("Unknown policy type")
    
    for ep in range(episodes):
        metrics = rollout_episode(env, policy_fn)
        results.append(metrics)

    return results

#aggregating results
def summarize(results):
    def mean(key):
        return np.mean([r[key] for r in results])

    return {
        "lap_completion_rate": mean("completed_lap"),
        "avg_steps": mean("steps"),
        "avg_reward": mean("total_reward"),
        "avg_lateral_error": mean("mean_lateral_error"),
        "avg_speed": mean("mean_speed"),
    }

if __name__ == "__main__":
    expert_results = evaluate("expert", episodes=20)
    bc_results = evaluate("bc", episodes=20)

    print("Expert summary:", summarize(expert_results))
    print("BC summary", summarize(bc_results))