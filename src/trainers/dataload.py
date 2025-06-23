import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset 

class DataLoad:
    def __init__(self,data_path,max_size,max_stage):
        self.data_path = data_path + "1-"
        self.max_size = max_size
        self.max_stage = max_stage
        self.size = 0
        self.stage = 1

    def get_data(self):
        return __load()

    def __load(self):
        observations = np.empty((0,224,256,3))
        actions = np.empty((0,6))
        path = self.data_path + str(self.stage) + "/mario_trajectory_j" + str(self.size) + ".npz"
        if self.size == self.max_size:
            self.size = 0
            self.stage +=1

        if self.stage == self.max_stage:
            print("data is over")
            return None,None

        if os.path.exists(path):
            print(f"stage {self.stage} のデータをロードしています...{self.size}")
            data = np.load(path)
            observations = np.append(observations,data['observations'],axis=0)
            actions = np.append(actions,data['actions'],axis=0)
            self.size +=1
            return observations,actions
        return None,None
