from .base_trainer import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import networks

class CFILTrainer(BaseTrainer):
    def __init__(self,learning_rate=0.001):
        super(CFILTrainer,self).__init__(learning_rate=learning_rate)
        self.state_encoder = networks.StateEncoder(input_size=3,representation_size=32,representation_final_size=35)
        self.action_encoder = networks.ActionEncoder(input_size=6,representation_size=16,output_size=3)
        self.state_transition_model = networks.StateTransitionModel(input_size=38,representation_size=64,output_size=35)
        all_trainable_parameters = list(self.state_encoder.parameters()) + list(self.action_encoder.parameters()) + list(self.state_transition_model.parameters())
        self.optimizer = optim.Adam(all_trainable_parameters, lr=self.learning_rate)

    def train(self,model,dataset,num_epoches=10,batch_size=32):
        mse_loss = nn.MSELoss()

        print("start training...")
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        for epoch in range(num_epoches):
            total_loss = 0
            predicted_state = torch.zeros(32,32)
            for batch_s_t_obs, batch_actions ,batch_s_t_plus_1_obs in dataloader:
                batch_s_t_obs = batch_s_t_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_s_t_plus_1_obs = batch_s_t_plus_1_obs.to(self.device)
                # Forward 
                t_state = self.state_encoder(batch_s_t_obs)
                t1_state = self.state_encoder(batch_s_t_plus_1_obs)
                action = self.action_encoder(batch_actions)
                predicted_state= self.state_transition_model(torch.cat((t_state,action),dim=1))
                #  Backward
                loss = mse_loss(predicted_state, t1_state)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print("total_loss: ",total_loss)
            # ゲームオーバーを理解する
