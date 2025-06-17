import os
import torch
import gymnasium as gym
import ppaquette_gym_super_mario

class Run:
    def __init__(self,network,agent,model_path):
        self.network = network
        self.agent = agent
        self.model_path = model_path

    def run(self,level):
        env = self.create_env(level)
        if env == None:
            print("env create error")
            return 

        model = self.load_model(self.model_path,network)
        if model == None:
            print("model load error")
            return
        self.agent.setting(env,model)

        traj,total_reward = agent.roolout()
        print("total_reward",total_reward)

    def create_env(self,level):
        if level != None:
            print("level:",level)
        else:
            print("please input level")
            return None
        env = gym.make('ppaquette/SuperMarioBros-'+level+'-v0')
        return env

    def load_model(self,path,network):
        if not os.path.exists(path):
            print(f"エラー: 学習済みモデル '{path}' が見つかりません。")
            print("先に 'train_bc_model.py' を実行してモデルを作成してください。")
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)
        network.load_state_dict(torch.load(path, map_location=device))
        network.eval()
        print("success load model")
        return network 
