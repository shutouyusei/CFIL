from .base_network import BaseNetwork
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import os

class MarioNetwork(BaseNetwork):
    def __init__(self):
        super(MarioNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), # H/2, W/2
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2), # H/4, W/4
                    
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2), # H/8, W/8

                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2) # H/16, W/16
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
        

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x

