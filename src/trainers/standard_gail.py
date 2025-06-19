import numpy as np
import networks
import agents
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class GAILTrainer(BaseTrainer):
    def __init__(self,model,dataset, config = {"learning_rate":0.001, 
                                               "discriminator":networks.Discriminator(172032,6,64), 
                                               "num_epoches":10, 
                                               "batch_size":32, 
                                               "policy_iterations":1,
                                               "discriminator_iterations":1}):
        super().__init__(model,dataset,config)
        self.__set_config(config)
        self.__setting()

    def __set_config(self,config):
        print(config)
        self.learning_rate = config["learning_rate"]
        self.discriminator = config["discriminator"]
        self.num_epoches = config["num_epoches"]
        self.batch_size = config["batch_size"]
        self.policy_iterations = config["policy_iterations"]
        self.discriminator_iterations = config["discriminator_iterations"]

    def __setting(self):
        self.discriminator.to(self.device)
        self.optimizer_policy = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

    def train(self):
        expert_obs_data = obs_data[:obs_data.shape[0]//2]
        agent_obs_data = obs_data[obs_data.shape[0]//2:]
        expert_action_data = action_data[:action_data.shape[0]//2]

        self.__setting(model,discriminator)
        dataset = TensorDataset(expert_obs_data, expert_action_data,agent_obs_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epoches):
            total_policy_loss = 0
            total_discriminator_loss = 0
            for _ in range(discriminator_iterations):
                total_discriminator_loss += self.train_discriminator(dataloader,agent_obs_data)
            for _ in range(policy_iterations):
                total_policy_loss += self.train_policy(dataloader)
            print(f"Epoch {epoch+1}/{num_epoches}, "
                  f"Discriminator Loss: {total_discriminator_loss:.4f}, "
                  f"Policy Loss: {total_policy_loss:.4f}")
        print("IRL Training complete.")
        return self.model

    def train_discriminator(self,data_loader,agent_obs_data):
        total_discriminator_loss = 0
        for exp_obs, exp_action,agent_obs in data_loader:
            # エキスパートは１に近づけるように学習
            prob_expert = self.discriminator(exp_obs, exp_action)
            loss_expert = F.binary_cross_entropy(prob_expert, torch.ones_like(prob_expert))

            agent_actions = self.model(agent_obs)
            agent_actions = torch.sigmoid(agent_actions)

            # エージェントは0に近づけるように学習
            prob_agent = self.discriminator(agent_obs.detach(), agent_actions.detach())
            loss_agent = F.binary_cross_entropy(prob_agent, torch.zeros_like(prob_agent))

            discriminator_loss = loss_expert + loss_agent
            self.optimizer_discriminator.zero_grad()
            discriminator_loss.backward()
            self.optimizer_discriminator.step()
            total_discriminator_loss += discriminator_loss.item()
        return total_discriminator_loss

    def train_policy(self,data_loader):
        total_loss = 0
        for exp_obs, exp_action,agent_obs in data_loader:

            agent_obs = exp_obs
            agent_actions = self.model(agent_obs)
            agent_actions = torch.sigmoid(agent_actions)

            prob_agent_for_policy = self.discriminator(agent_obs, agent_actions)
            policy_reward = -torch.log(prob_agent_for_policy + 1e-8)

            policy_loss = policy_reward.mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
            total_loss += policy_loss.item()
        return total_loss
