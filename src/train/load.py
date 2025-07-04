import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset 

class Load:
    def load(self,size=1,stage=1):
        obs,action = self.__load_data(size,stage)
        obs,action = self.__to_tensor(obs,action)
        return TensorDataset (obs,action)

    def __to_tensor(self,obs,action):
        obs_data = torch.from_numpy(obs).float() / 255.0
        obs_data = obs_data.permute(0, 3, 1, 2)
        act_data = torch.from_numpy(action).type(torch.float32)
        print("finish to tensor")
        return obs_data,act_data

    def __load_data(self,size=1,stage=1,is_success=True):
        DATA_DIR ="../data/"
        if is_success:
            DATA_DIR = "../data/success/"
        else:
            DATA_DIR = "../data/failure/"

        SAMPLE_SIZE = 500
        num_of_file = stage * size
        observations = np.empty((num_of_file,SAMPLE_SIZE,224,256,3))
        actions = np.empty((num_of_file,SAMPLE_SIZE,6))
        for level in range(1,stage+1):
            for num in range(size):
                data_path = DATA_DIR +"1-"+ str(level) + "/mario_trajectory" + "_" + str(num) + ".npz"
                if os.path.exists(data_path):
                    print(f"stage {level} のデータをロードしています...{num}")
                    data = np.load(data_path)
                    index = (level - 1 ) * size + num
                    observations[index] = data['observations']
                    actions[index] = data['actions']
                    num +=1

        reshaped_obs = observations.reshape(observations.shape[0] * observations.shape[1],observations.shape[2],observations.shape[3],observations.shape[4])
        reshaped_act = actions.reshape(actions.shape[0] * actions.shape[1],actions.shape[2])
        print("finish load")
        return reshaped_obs,reshaped_act

