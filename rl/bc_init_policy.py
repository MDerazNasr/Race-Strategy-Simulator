"""
Key idea
your BC policy is an MLP
-- input dim = obs (should now be 6)
-- output dim = 2 (throttle, steer)

SB3's MLPolicy is also an MLP but structured slightly diffrently:
    - it has a shared feature extractor + policy/value heads

So we cannot just load state_dict directly
- we'll copy the relevant layers

THE BEST WAY
- build your BC network inside a custom SB3 policy
so weight transfer is exact
"""

# Custom SB3 policy that matches your BC network
import sys
from pathlib import Path

import torch as th
import torch.nn as nn

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3.common.policies import ActorCriticPolicy


# subclassing SB3's built in policy
# making PPO use your class instead of MlpPolicy
# what am i doing in this following class
class BCInitPolicy(ActorCriticPolicy):
    """
    Custom policy whose actor network matches our BC MLP:
        6 -> 128 -> 128 -> 2 with Tanh
    Plus a value network for PPO

    actor matches BC
    critic predicts value for PPO
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CONTINUE THE REST


"""
You want PPO to start from your Behavior Cloning (BC) actor weights.

In SB3 PPO, the policy is an actor-critic:

Actor: outputs an action distribution (for continuous actions: usually Gaussian)

Critic: outputs a value V(s) (expected return)

SB3’s ActorCriticPolicy already provides:

feature extraction (extract_features)

action distribution handling (sample actions, compute log probs, entropy)

value network head (value_net) usually

correct shapes + training hooks

You are overriding to:

make actor architecture match your BC MLP: 6 -> 128 -> 128 -> 2

add a critic network of similar size

That’s a good goal.

But: SB3 PPO requires valid log-probs from the action distribution. Returning zeros breaks PPO’s learning math.

I’ll explain your code, then show what needs fixing.

"""
