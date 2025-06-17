import numpy as np
import torch
import gymnasium as gym

class BaseAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initalize(self, env,model):
        self.env = env
        self.model = model
        
    def select_action(self,state):
        raise NotImplementedError

    def perform_action(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        return obs,reward,done

    def roolout(self):
        state, info = self.env.reset()
        traj = []
        total_reward = 0

        done = False

        while not done:
            action = self.select_action(state)
            next_state,reward,done = self.perform_action(action)
            traj += [(state,action)] 
            total_reward += reward

            state = next_state 
        return traj, total_reward

