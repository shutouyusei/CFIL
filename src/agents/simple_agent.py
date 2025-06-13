import torch
from .base_agent import BaseAgent

class SimpleAgent(BaseAgent):
    def __init__(self, env,model):
        super(SimpleAgent, self).__init__(env)
        self.model = model

    def select_action(self,state):
        obs = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        obs = obs.permute(0,3,1,2)
        with torch.no_grad():
            action_logists = self.model(obs)
            action_probs = torch.sigmoid(action_logists)
            predicted_action = (action_probs > 0.5).int().cpu().numpy().squeeze(0)
        return predicted_action.tolist()
