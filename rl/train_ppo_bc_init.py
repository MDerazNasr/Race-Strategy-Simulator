# Correct approach - SB3 net_arch + weight mapping
# step 1 - forve SB3 to use the same architecture
# SB3 can build actor/critic MLPs with the same hidden sizes using policy_kwargs

import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bc.train_bc import BCPolicy
from rl.make_env import make_env


def load_bc_into_ppo(ppo_model, bc_path, device):
    """
    copies BC MLP weights into PPO actor network

    BCPolicy: Linear(6,128) -> Linear(128,128) -> Linear(128,2)
    PPO actor has similar layers under: model.policy.mlp_extractor.policy_net + action_net
    """

    # Load BC
    bc = BCPolicy(state_dim=6, action_dim=2).to(device)
    bc.load_state_dict(torch.load(bc_path, map_location=device))
    bc.eval()

    # SB3 internals
    policy_net = ppo_model.policy.mlp_extractor.policy_net  # [Linear, Act, Linear, Act]
    action_net = ppo_model.policy.action_net  # final Linear to action dim

    # BC internals
    # bc.net [Linear, ReLU, Linear, ReLU, Linear, Tanh]
    bc_layers = [m for m in bc.net if hasattr(m, "weight")]

    # Map weights layer-by-layer:
    #
    #   BC structure (bc.net):
    #     [0] Linear(6, 128)    ← hidden layer 1 weights
    #     [1] ReLU
    #     [2] Linear(128, 128)  ← hidden layer 2 weights
    #     [3] ReLU
    #     [4] Linear(128, 2)    ← output weights (throttle, steer)
    #     [5] Tanh
    #
    #   SB3 PPO actor structure:
    #     policy_net[0] = Linear(6, 128)    ← matches bc_layers[0]
    #     policy_net[1] = ReLU
    #     policy_net[2] = Linear(128, 128)  ← matches bc_layers[1]
    #     policy_net[3] = ReLU
    #     action_net    = Linear(128, 2)    ← matches bc_layers[2]
    #
    # NOTE: BC has Tanh on the output; SB3 applies tanh via its action distribution
    # bijector (squashing the Gaussian sample). The linear weights transfer exactly.
    with torch.no_grad():
        # Hidden layer 1
        policy_net[0].weight.copy_(bc_layers[0].weight)
        policy_net[0].bias.copy_(bc_layers[0].bias)

        # Hidden layer 2
        policy_net[2].weight.copy_(bc_layers[1].weight)
        policy_net[2].bias.copy_(bc_layers[1].bias)

        # Output layer — THIS WAS MISSING: without this, the action head is
        # random despite the BC warm start. The agent had a good trunk but a
        # random head, so it would produce completely wrong actions at startup.
        action_net.weight.copy_(bc_layers[2].weight)
        action_net.bias.copy_(bc_layers[2].bias)

    print("[BC Init] Copied all 3 BC layers into PPO actor. Weight transfer complete.")


def train():
    env = DummyVecEnv([make_env])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device=device,
    )
    model.set_logger(configure("runs/ppo_bc_init", ["stdout", "tensorboard"]))

    # Load BC weights into PPO actor
    load_bc_into_ppo(model, "bc/bc_policy_final.pt", device)
    # Train
    model.learn(total_timesteps=300_000)
    # Save
    model.save("rl/ppo_bc_init.zip")


if __name__ == "__main__":
    train()
