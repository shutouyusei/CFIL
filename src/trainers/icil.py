import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from .base_trainer import BaseTrainer

class ICILTrainer(BaseTrainer):
    def __init__(self, model,dataset,config):
        super(ICILTrainer, self).__init__(model,dataset,config)

    def _set_config(self):
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.num_epoches = self.config["num_epoches"]

    def _train_model(self):


    def __learn_invariant_representation(self):

