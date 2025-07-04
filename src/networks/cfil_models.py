from .base_network import BaseNetwork
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import os

class RepresentationNet(BaseNetwork):
    def __init__(self):
        super(RepresentationNet, self).__init__()

class EnvironmentClassfierNet(BaseNetwork):
    def __init__(self):
        super(EnvironmentClassfierNet, self).__init__(
