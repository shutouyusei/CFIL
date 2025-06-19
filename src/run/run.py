import os
import torch
import gymnasium as gym
from pathlib import Path
import ppaquette_gym_super_mario

class Run:
    def __init__(self,network,agent_factory,model_path):
        self.network = network
        self.agent_factory = agent_factory
        current_file_path = Path(__file__).resolve()
        parent_dir = current_file_path.parent.parent.parent
        self.model_path = parent_dir/model_path

    def run(self,level):
        env = self.__create_env(level)
        if env == None:
            print("env create error")
            return 

        model = self.__load_model(self.model_path,self.network)
        if model == None:
            print("model load error")
            return
        agent = self.agent_factory.create(model,env)

        traj,total_reward = agent.roolout()
        print("total_reward",total_reward)

    def __create_env(self,level):
        if level != None:
            print("level:",level)
        else:
            print("please input level")
            return None
        env = gym.make('ppaquette/SuperMarioBros-'+level+'-v0')
        return env

    def __load_model(self,path,network):
        if not os.path.exists(path):
            print(f"エラー: 学習済みモデル '{path}' が見つかりません。")
            print("先に 'train_bc_model.py' を実行してモデルを作成してください。")
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)
        network.load_state_dict(torch.load(path, map_location=device))
        network.eval()
        print(path)
        return network 
