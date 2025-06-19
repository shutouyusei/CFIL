import sys
import agents
import networks
from .run import Run

network = networks.BaseNetwork()
agent_factory = agents.AgentFactory()

if sys.argv[2] == '1':
    network = networks.PolicyNet(3,6,32)
    agent_factory = agents.SimpleAgentFactory()
    run = Run(network,agent_factory,"data/mario_bc_model.pth")
    run.run(sys.argv[1])
else:
    network = networks.PolicyNet(3,6,32)
    agent_factory = agents.SimpleAgentFactory()
    run = Run(network,agent_factory,"data/mario_gail_model.pth")
    run.run(sys.argv[1])
