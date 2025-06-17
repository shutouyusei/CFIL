import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class GAILTrainer(BaseTrainer):
    def __init__(self,learning_rate=0.001,gamma=0.99, gail_lambda=0.1):
        super().__init__(learning_rate)

        self.gamma = gamma
        self.gail_lambda = gail_lambda

    def setting(self,policy_network,discriminator):
        self.policy_network = policy_network
        self.discriminator = discriminator
        self.policy_network.to(self.device)
        self.discriminator.to(self.device)
        self.optimizer_policy = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

    def train(self,policy_network,discriminator, obs_data,action_data, num_epoches=10,batch_size=32,policy_iterations=1,discriminator_iterations=1):
        self.setting(policy_network,discriminator)
        dataset = TensorDataset(obs_data, action_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epoches):
            total_policy_loss = 0
            total_discriminator_loss = 0
            for _ in range(discriminator_iterations):
                total_discriminator_loss += self.train_discriminator(dataloader,dataset)
            for _ in range(policy_iterations):
                total_policy_loss += self.train_policy(dataloader,batch_size)
            print(f"Epoch {epoch+1}/{num_epoches}, "
                  f"Discriminator Loss: {total_discriminator_loss:.4f}, "
                  f"Policy Loss: {total_policy_loss:.4f}")
        print("IRL Training complete.")
        return self.policy_network

    def random_sample(self,dataset,size):
        indices  = torch.randint(0, len(dataset),size)
        agent_obs_batch = dataset['observations'][indices]
        agent_action_logits = self.policy_network(agent_obs_batch)
        agent_action_batch = torch.sigmoid(agent_action_logits)  
        agent_action_batch = (agent_action_batch > 0.5).int().cpu().numpy().squeeze(0)
        return agent_obs_batch, agent_action_batch

    def train_discriminator(self,data_loader,dataset):
        total_discriminator_loss = 0
        for exp_obs, exp_action in data_loader:
            self.optimizer_discriminator.zero_grad()
            # エキスパートは１に近づけるように学習
            prob_expert = self.discriminator(exp_obs, exp_action)
            loss_expert = F.binary_cross_entropy(prob_expert, torch.ones_like(prob_expert))

            agent_obs, agent_actions = self.random_sample(dataset, exp_obs.size(0))
            # エージェントは0に近づけるように学習
            prob_agent = self.discriminator(agent_obs.detach(), agent_actions.detach())
            loss_agent = F.binary_cross_entropy(prob_agent, torch.zeros_like(prob_agent))

            discriminator_loss = loss_expert + loss_agent
            discriminator_loss.backward()
            self.optimizer_discriminator.step()
            total_discriminator_loss += discriminator_loss.item()
        return total_discriminator_loss

    def train_policy(self,data_loader,batch_size):
        self.optimizer_policy.zero_grad()
        total_loss = 0
        for exp_obs, exp_action in data_loader:

            agent_obs = exp_obs
            agent_actions_logits = self.policy_network(agent_obs)
            agent_actions = agent_actions_logits

            prob_agent_for_policy = self.discriminator(agent_obs, agent_actions)
            policy_reward = -torch.log(prob_agent_for_policy + 1e-8)

            policy_loss = policy_reward.mean()

            policy_loss.backward()
            self.optimizer_policy.step()
            total_loss += policy_loss.item()
        return total_loss
