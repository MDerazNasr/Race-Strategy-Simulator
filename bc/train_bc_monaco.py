"""
Behavioral Cloning on Monaco Circuit — D42.

Identical architecture and loss to bc/train_bc.py, trained on Monaco
expert demonstrations from expert/collect_data.generate_monaco_dataset().

Architecture: 11D obs → [128 → ReLU → 128 → ReLU] → 2D action
Loss: MSE (mean squared error) between predicted and expert actions
Epochs: 200, batch_size=256, Adam lr=1e-3 with cosine annealing

Saves: bc/bc_policy_monaco.pt (weights only, compatible with bc_init_policy.py)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ExpertDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.states  = data["states"]
        self.actions = data["actions"]
        assert self.states.shape[0] == self.actions.shape[0]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
        )


class BCPolicy(nn.Module):
    def __init__(self, state_dim: int = 11, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def train_bc_monaco(
    data_path:   str = "bc/expert_data_monaco.npz",
    output_path: str = "bc/bc_policy_monaco.pt",
    epochs:      int = 200,
    batch_size:  int = 256,
    lr:          float = 1e-3,
    device:      str = "cpu",
):
    data_file = Path(project_root) / data_path
    if not data_file.exists():
        raise FileNotFoundError(
            f"{data_file} not found.\n"
            "Run: from expert.collect_data import generate_monaco_dataset; generate_monaco_dataset()"
        )

    dataset    = ExpertDataset(str(data_file))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    state_dim  = dataset.states.shape[1]   # 11
    action_dim = dataset.actions.shape[1]  # 2
    print(f"[BC Monaco] Dataset: {len(dataset)} samples, state_dim={state_dim}, action_dim={action_dim}")

    policy    = BCPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    print(f"[BC Monaco] Training {epochs} epochs, batch={batch_size}, lr={lr}")

    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0
        for states_b, actions_b in dataloader:
            states_b  = states_b.to(device)
            actions_b = actions_b.to(device)
            pred = policy(states_b)
            loss = criterion(pred, actions_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/n_batches:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    out_file = Path(project_root) / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(out_file))
    print(f"[BC Monaco] Saved → {out_file}")
    return str(out_file)


if __name__ == "__main__":
    train_bc_monaco()
