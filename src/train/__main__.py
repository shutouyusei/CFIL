import trainers
import networks
import torch

MODEL_SAVE_PATH = "mario_bc_model.pth"
trainer = trainers.BCTrainer(learning_rate=0.001)
model = networks.MarioNetwork() 
dataset = data.load_data()
model = trainer.train(model,dataset)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"モデルを '{MODEL_SAVE_PATH}' に保存しました。")
