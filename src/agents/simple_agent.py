import torch
from .base_agent import BaseAgent,AgentFactory

class SimpleAgent(BaseAgent):
    def select_action(self,state):
        obs = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        obs = obs.permute(0,3,1,2)
        with torch.no_grad():
            action_logits = self.model(obs)
            action_logits = torch.sigmoid(action_logits)
            predicted_action = (action_logits> 0.5).int().cpu().numpy().squeeze(0)
        return predicted_action.tolist()

class SimpleAgentFactory(AgentFactory):
    def _create_agent(self,model):
        return SimpleAgent(model)
