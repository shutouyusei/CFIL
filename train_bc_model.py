from net import MarioBCModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import os

class Trainer:
    def __init__(self,learning_rate=0.001):
        num_actions = 6
        self.model = MarioBCModel(num_actions)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"使用デバイス: {device}")
        pos_weight = torch.tensor([1.0, 1.0, 5.0, 1.0, 1.0, 1.0])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self):
        observations = np.empty((0,224,256,3))
        actions = np.empty((0,6))
        for level in range(4):
            num = 0
            while True:
                data_path = "data/1-"+str(level)+"/mario_trajectory_"+str(num)+".npz"
                if not os.path.exists(data_path):
                    print(f"ERROR: '{data_path}' is not finded")
                    break

                print(f"load '{data_path}'...")
                loaded_data = np.load(data_path)
                observations = np.append(observations, loaded_data['observations'], axis=0)
                actions = np.append(actions, loaded_data['actions'], axis=0)
                num += 1
            obs_data = torch.from_numpy(observations).float() / 255.0
            actions_data = torch.from_numpy(actions).type(torch.float32)

        return obs_data,actions_data

    def train(self,obs_data,actions_data,num_epoches=10,batch_size=32):
        print("start training...")
        dataset = TensorDataset(obs_data, actions_data)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        for epoch in range(num_epoches):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                device = next(self.model.parameters()).device
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                # Forward 
                predicted_action = self.model(batch_obs)
                loss = self.criterion(predicted_action, batch_actions)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epoches}], Loss: {avg_loss:.4f}")
        print("training is completed.")

    def get_model(self):
        return self.model

if __name__ == "__main__":
    MODEL_SAVE_PATH = "mario_bc_model.pth"
    trainer = Trainer(learning_rate=0.001)
    obs_data,actions_data = trainer.load_data()
    trainer.train(obs_data,actions_data)
    model = trainer.get_model()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"モデルを '{MODEL_SAVE_PATH}' に保存しました。")
