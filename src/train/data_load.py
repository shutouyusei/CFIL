import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset 

DATA_DIR ="../data/"

class Data:
    def load(self,size,stage):
        obs,action = self.load_data(size,stage)
        return TensorDataset(obs,action)

    def load_data(self,size,stage=1):
        observations = np.empty((0,224,256,3))
        actions = np.empty((0,6))
        for level in stage:
            for num in range(size):
                data_path = DATA_DIR +"1-"+ str(level) + "/mario_trajectory" + "_" + str(num) + ".npz"
                if os.path.exists(data_path):
                    print(f"stage {level} のデータをロードしています...{num}")
                    data = np.load(data_path)
                    observations = np.append(observations,data['observations'],axis=0)
                    actions = np.append(actions,data['actions'],axis=0)
                    num +=1
                else:
                    print(f"level {level} のデータをロードしました。")
                    break

        obs_data = torch.from_numpy(observations).float() / 255.0
        obs_data = obs_data.permute(0, 3, 1, 2)
        act_data = torch.from_numpy(actions).type(torch.float32)

        return obs_data,act_data
