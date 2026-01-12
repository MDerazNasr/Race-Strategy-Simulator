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
):
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
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate) #adam optimizer updates weights to reduce loss, lr is learning rate.
    criterion = nn.MSELoss() #mean squared error loss -> compares predicted action v expert action

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
            #loss
            loss = criterion(pred_actions, batch_actions)
            #backward pass, clears old gradients, imp because pytorch accumulates gradients by default
            loss.backward() #backpropogation loss for all weight
            optimizer.step() #adam updates parametres using gardients

            #trackingh loss for printing
            epoch_loss += loss.item() #converts a 1-element tensor into a python float
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}") #average loss per batch, print progress

    #save trained model
    torch.save(policy.state_dict(), save_path)
    print(f"Saved BC policy to save {save_path}")

if __name__ == "__main__":
    train_bc()