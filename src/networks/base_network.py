import torch
import torch.nn as nn
import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def forward(self,x):
        raise NotImplementedError
