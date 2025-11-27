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

def run_expert_lap():
    env = F1Env()
    expert = ExpertDriver(env.track)

    obs, info = env.reset()
    xs,ys = [], []

    for step in range(2000):
        action = expert.get_action(env.car)
        obs, reward, terminated, truncated, info = env.step(action)

        xs.append(env.car.x)
        ys.append(env.car.y)

        if terminated:
            print("Expert went off track")
            break
            
    plt.plot(xs, ys, label="Expert path")
    plt.plot(env.track[:,0], env.track[:,1], '--', alpha=0.5, label="Track")
    plt.axis('equal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_expert_lap()

'''
What this gives you:
	A line plot showing if the expert car can circle the track
	if the car stays on the centerline → physics + steering logic works
	If it oscillates → tune lookahead, max_speed, corner_factor

Now your environment has a working, deterministic human-designed driver.
'''