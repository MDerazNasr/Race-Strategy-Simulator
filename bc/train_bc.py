"""Behavioral cloning training script.

behvaviour cloning in simple supervised learning

this script:
- loads states and actions from a .npz file
- wraps them in a pytorch dataset so you can batch them
- defines a simple MLP policy network (BCPolicy)
- trains it with MSE loss: predicted actions vs expert actions
- saves the trained weights to disk for later use in rollouts
"""

import numpy as np #to load .npz file
import torch
import torch.nn as nn
import torch.optim as optim #contains optimizers
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
#Dataset - class interface for a collection of training examples
#dataLoader - helper that makes batches, shuffles and iterates efficiently
from torch.utils.data import Dataset, DataLoader

class ExpertDataset(Dataset):
    def __init__(self, npz_path: str):
        """
        Loads states and actions from a .npz file saved earlier.

        npz file must contain:
        - 'states': shape (N, 6)
        - 'actions': shape (N, 2)
        """
        data = np.load(npz_path)
        self.states = data["states"] #numpy array, float32
        self.actions = data["actions"] #numpy array, float32
        
        '''
        assert condition, message means -> if condition is false -> crash with message
        self.states.shape[0] = number of rows = number of samples
        backslash means continue this line onto next 
        '''
        assert self.states.shape[0] == self.actions.shape[0], \
            "States and actions must have same length"
        
    def __len__(self):
        # number of samples
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        state = self.states[idx] #shape(6,)
        action = self.actions[idx] #shape (2, )

        state = torch.from_numpy(state) #torch.Size([7])
        action = torch.from_numpy(action) #torch.Size([2])
        action = torch.clamp(action, -1.0, 1.0)

        return state, action

#Your policy is a neural network module
class BCPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        super().__init__()
        #like a pipeline, input goes in first layer, output of that foes into the next
        self.net = nn.Sequential(
            #fully connected layer
            nn.Linear(state_dim, hidden_dim),
            #activation function, adds nonlinearlity so netwoirk can learn complex patters
            nn.ReLU(),
            #maps hidden features to hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #final layer: maps to action output, output size 2
            nn.Linear(hidden_dim, action_dim),
            #squashes output into [-1,1], matches your env actions bounds, good when actions are normalized
            nn.Tanh(), #Keeping outputs in [-1,1]
        )
    
    def forward(self, state):
        '''
        state - tensor of shape (batch_size, state_dim)
        returns: tensor of shape (batch_size, action_dim) 
        
        during training you feed batches, not single samples

        if batch size is B:
            input: (B, 6)
            output: (B, 2)
        '''
        return self.net(state)
        
#training loop
def train_bc(
    npz_path="bc/expert_data.npz", #where expert data is
    num_epochs=20, #how many times you loop through the dataset
    batch_size=256, #how many samples per gradient step
    learning_rate=1e-3, #step size for optimizer
    device=None, #cpu/gpu choice
    save_path="bc/bc_policy.pt", #where to save weights
    pit_dim=None,          # action dimension index for pit signal (None = no weighting)
    pit_class_weight=1.0,  # weight multiplier for pit-positive samples (d20 Fix A)
):
    """
    Train a BC policy via supervised imitation learning.

    Standard usage (d1-d19): no weighting, MSE loss over all samples equally.

    Pit-weighted usage (d20 Fix A):
      The pit-stop dataset has extreme class imbalance: ~0.10% pit-positive
      samples. Standard MSE ignores the minority class — the gradient signal
      from 86 pit-positive samples is diluted by 86,752 pit-negative samples.

      Setting pit_dim=2 and pit_class_weight=1000 upweights pit-positive
      samples 1000x, making their gradient contribution equivalent to 86,000
      samples. This gives approximately balanced learning:
        Effective pit-positive:  86 × 1000 = 86,000
        Effective pit-negative:  86,752 × 1 = 86,752
        Ratio:                   ~1:1  (was 1:1009 before weighting)

      The BC network now receives a meaningful gradient from pit-positive
      samples and should learn: tyre_life < 0.3 → pit_signal → +1.0.

    Args:
        pit_dim:          Index of pit_signal in the action vector (e.g. 2 for [throttle,steer,pit]).
                          None = backward-compatible mode, no weighting.
        pit_class_weight: Multiplier for pit-positive sample losses.
                          1.0 = uniform (standard BC).
                          1000.0 = 1000x upweight for pit-positive (d20).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

    #create dataset and dataloader
    dataset = ExpertDataset(npz_path)
    dataloader = DataLoader(
        dataset, #creates batches and iterates
        batch_size=batch_size, #group samples into batches of 256
        shuffle=True, #randomise ordering each epoch (imp)
        drop_last=True, #if final batch is smaller than 256, drop it --> keeps batch sizes consistent (not always required)
    )

    #Initialise policy network
    state_dim = dataset.states.shape[1] #should be 6
    action_dim = dataset.actions.shape[1] #should be 2
    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim).to(device) #creates the network and moves it cpu/gpu

    #optimiser and loss
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Use reduction='none' so we can apply per-sample weights before averaging.
    # For standard BC (pit_dim=None), this is mathematically identical to MSELoss().
    criterion = nn.MSELoss(reduction='none')

    # Log whether pit weighting is active
    use_pit_weight = (pit_dim is not None and pit_class_weight > 1.0)
    if use_pit_weight:
        pit_positive_count = int((dataset.actions[:, pit_dim] > 0).sum())
        print(f"[BC] Pit-weighted training: pit_dim={pit_dim}, weight={pit_class_weight}x")
        print(f"     Pit-positive samples: {pit_positive_count} / {len(dataset)} "
              f"({100*pit_positive_count/len(dataset):.2f}%)")
        print(f"     Effective ratio after weighting: "
              f"{pit_positive_count*pit_class_weight:.0f} vs {len(dataset)-pit_positive_count}")

    #Training loop (core learning)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_states, batch_actions in dataloader:
            #imp - model and data must be on same device
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            #forward pass
            pred_actions = policy(batch_states)

            # Per-element MSE loss: shape (batch_size, action_dim)
            per_elem_loss = criterion(pred_actions, batch_actions)
            # Average over action dimensions -> per-sample loss: shape (batch_size,)
            per_sample_loss = per_elem_loss.mean(dim=1)

            # Apply pit class weighting if configured (d20 Fix A).
            # pit-positive samples (where action[pit_dim] > 0) are upweighted.
            # This counteracts the 1:1009 class imbalance in the pit dataset.
            if use_pit_weight:
                pit_positive = (batch_actions[:, pit_dim] > 0).float()
                # Weight = 1.0 for non-pit samples, pit_class_weight for pit samples
                sample_weights = 1.0 + (pit_class_weight - 1.0) * pit_positive
                per_sample_loss = per_sample_loss * sample_weights

            mse_loss = per_sample_loss.mean()
            action_penalty = 0.01 * torch.mean(pred_actions ** 2)
            loss = mse_loss + action_penalty

            #backward pass, clears old gradients, imp because pytorch accumulates gradients by default
            loss.backward() #backpropogation loss for all weight
            optimizer.step() #adam updates parametres using gardients
            optimizer.zero_grad() #clear gradients for next iteration

            #trackingh loss for printing
            epoch_loss += loss.item() #converts a 1-element tensor into a python float
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}") #average loss per batch, print progress

    #save trained model
    torch.save(policy.state_dict(), save_path)
    print(f"Saved BC policy to {save_path}")

    return policy, save_path

if __name__ == "__main__":
    policy, save_path = train_bc()
    
    torch.save(policy.state_dict(), "bc/bc_policy_final.pt")
    print(f"Saved final BC policy to bc/bc_policy_final.pt")

'''
Mental Model (Remember This)
    train_bc.py → creates the policy
    bc_policy_final.pt → frozen artifact
    RL training → consumes this artifact
    Think of it like a pretrained ImageNet model.

'''