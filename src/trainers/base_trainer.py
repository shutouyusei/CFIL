import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

class BaseTrainer:
    def __init__(self,model=None,dataset=None,config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.dataset = dataset 
        self.config = config
        print(self.config)

    def __set_config(self):
        raise NotImplementedError

    def _train_model(self):
        raise NotImplementedError

    def train(self):
        print("set config")
        self._set_config()
        print("start training...")
        total_train_loss  = self._train_model()
        print("loss:",total_train_loss)
        print("--- Training Completed ---")
        return self.model




