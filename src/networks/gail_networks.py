import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, obs_dim,action_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, obs, action):
        obs_flattened = obs.view(obs.size(0), -1)
        x = torch.cat([obs_flattened, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
