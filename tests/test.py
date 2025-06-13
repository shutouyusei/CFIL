import trainers
import networks
import data
import torch

MODEL_SAVE_PATH = "mario_bc_model.pth"
trainer = trainers.CFILTrainer(learning_rate=0.001)
model = networks.MarioNetwork() 
dataset = data.load_cfil_data(11)
model = trainer.train(model,dataset)


