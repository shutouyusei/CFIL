import numpy as np
import torch
import os

class BaseTrainer:
    def __init__(self,learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

    def train(self,model,num_epoches=10,batch_size=32):
        raise NotImplementedError


