import numpy as np
import torch
import gymnasium as gym

class BaseAgent:
    def __init__(self,model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def select_action(self,state):
        raise NotImplementedError

    def roolout(self):
        if self.env is None:
            raise ValueError("env is not set")

        state, info = self.env.reset()
        traj = []
        total_reward = 0

        done = False

        while not done:
            action = self.select_action(state)
            next_state,reward,done = self.__perform_action(action)
            traj += [(state,action)] 
            total_reward += reward

            state = next_state 
        return traj, total_reward

    def __perform_action(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        return obs,reward,done


class AgentFactory:
    def create(self,model,env=None):
        agent = self._create_agent(model)
        if env is not None:
            agent.env = env
        return agent

    def _create_agent(self,model):
        raise NotImplementedError

