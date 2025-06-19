from .load import Load
from pathlib import Path
import torch
import networks
import trainers
import sys
import time


def get_trainers(setting):
    load = Load()
    if setting == '1':
        model = networks.PolicyNet(3,6,32) 
        config = {"learning_rate":0.001,"batch_size":32,"num_epoches":10}
        dataset = load.load(10,[1])
        return trainers.BCTrainer(model,dataset,config)

    elif setting == '2':
        trainer = trainers.GAILTrainer(learning_rate=0.001,gamma=0.99,gail_lambda=0.1)
        obs_data, action_data = load.__load_data(10,[1])
        policy_network = networks.PolicyNet(3,6,32)
        discriminator = networks.Discriminator(172032,6,64)
        return trainer

def get_path(setting):
    parent_path = Path(__file__).parent.parent
    if setting == 1:
        return  str(parent_path / Path("data/mario_bc_model.pth"))
    elif setting == '2':
        return str(parent_paht / Path("data/mario_gail_model.pth"))

trainers = get_trainers(sys.argv[1])
path = get_path(sys.argv[1])
print(path)

start_time = time.time()

model = trainers.train()

end_time = time.time()
elapsed_time = end_time - start_time

torch.save(model.state_dict(), path)

print(f"モデルを '{path}' に保存しました。")
print(f"実行時間: {elapsed_time:.2f} 秒")
