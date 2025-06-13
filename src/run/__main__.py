from .run_model import RunModel
import agents
import networks
import sys

MODEL_PATH = "../mario_bc_model.pth"

# main

run = RunModel()
args = sys.argv
env = run.create_env(args[1])
if env == None:
    print("env create error")
    exit()

# setting
network = networks.MarioNetwork()

model = run.load_model(MODEL_PATH,network)
if model == None:
    print("model load error")
    exit()

# setting
agent = agents.SimpleAgent(env,model)

traj,total_reward = agent.roolout()
print("total_reward",total_reward)
