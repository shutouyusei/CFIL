import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, input_size,output_size,hidden_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(input_size, hidden_size,kernel_size=5,stride=2,padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
        )
        def _get_conv_output_size(conv_model,input_shape):
            dummy_input = torch.rand(1,*input_shape)
            output = conv_model(dummy_input)
            return int(np.prod(output.shape))

        dummy_input_h = 224
        dummy_input_w = 256
        dummy_input_c = 3
        self.fc_input_size = _get_conv_output_size(self.conv_layers,(dummy_input_c,dummy_input_h,dummy_input_w))

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,6)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x

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
