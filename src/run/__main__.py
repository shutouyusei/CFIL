import sys
import agents
import networks
from .run import Run

network = networks.PolicyNetwork(3,6,64)
agent = agents.GAILAgent()
run = Run(network,agent,"../data/mairo_gail_model.pth")
run.run(sys.argv[1])
