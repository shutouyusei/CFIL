import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset 

def load(stage=1,index=0):
    DATA_DIR ="../data/"
    SAMPLE_SIZE = 500
    observations = np.empty((SAMPLE_SIZE,224,256,3))
    actions = np.empty((SAMPLE_SIZE,6))
    data_path = DATA_DIR +"1-"+ str(stage) + "/mario_trajectory" + "_" + str(index) + ".npz"
    if os.path.exists(data_path):
        print(f"stage {stage} のデータをロードしています...")
        data = np.load(data_path)
        observations = data['observations']
    else:
        print(f"level {stage} のデータをロードしました。")
        return None
    observations = torch.from_numpy(observations)
    return observations
