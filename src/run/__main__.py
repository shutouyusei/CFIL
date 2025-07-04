import sys
import agents
import networks
from .run import Run
import play

network = networks.BaseNetwork()
agent_factory = agents.AgentFactory()
save_path = "../data/movie"

if sys.argv[2] == '1':
    network = networks.PolicyNet(3,6,32)
    agent_factory = agents.SimpleAgentFactory()
    run = Run(network,agent_factory,"data/mario_bc_model.pth")
    reward = 0
    for i in range(10):
        trajectory,total_reward = run.run(sys.argv[1])
        reward += total_reward
    print("aberage reward:",reward/10))
else:
    network = networks.PolicyNet(3,6,32)
    agent_factory = agents.SimpleAgentFactory()
    run = Run(network,agent_factory,"data/mario_gail_model.pth")
    trajectory = run.run(sys.argv[1])
