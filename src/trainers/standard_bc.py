import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class BCTrainer(BaseTrainer):

    def __init__(self,model,dataset,config={"learning_rate":0.001,
                                            "batch_size":32,
                                            "num_epoches":10}):
        super(BCTrainer,self).__init__(model,dataset,config)

    def _set_config(self):
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.num_epoches = self.config["num_epoches"]

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        pos_weight = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.criterion.to(self.device)


    def _train_model(self):
        dataloader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)

        for epoch in range(self.num_epoches):
            total_val_loss = self.__bc_train(dataloader)
        return total_val_loss

    def __bc_train(self,dataloader):
        self.model.train()
        total_train_loss = 0
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(self.device)
            
            predicted_action_logits = self.model(batch_obs)
            
            batch_actions = batch_actions.float() 
            batch_actions = batch_actions.to(self.device)
            
            loss = self.criterion(predicted_action_logits, batch_actions)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(dataloader)
        return avg_train_loss
