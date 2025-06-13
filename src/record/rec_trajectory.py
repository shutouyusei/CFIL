import gymnasium as gym
from .key_setting import KeySetting
import numpy as np
from env.mario_env import MarioEnv
from ppaquette_gym_super_mario.wrappers.control import SetPlayingMode
from pynput import keyboard
import random
import os

class RecTrajectory:
    def __init__(self):
        self.key_setting = KeySetting()


    def start_rec(self,level):
        self.env = MarioEnv.create_env(level)
        self.key_setting.start_listen()
        self.env.reset()

        all_observations = []
        all_actions = []

        while True:
            actions = self.key_setting.get_actions()
            obs, reward, terminated, truncated, info = self.env.step(actions)
            if terminated:
                break

            all_observations.append(obs)
            all_actions.append(actions)
        self.key_setting.stop_listen()
        self.env.close()
        return all_observations, all_actions

class SaveTrajectory:
    def save(self,level,all_observations,all_actions):
        if all_observations and all_actions:
            print(f"\nCollected {len(all_observations)} frames of data.")
            # リストをNumPy配列に変換
            obs_trajectory = np.array(all_observations)
            actions_trajectory = np.array(all_actions)

            # データを .npz ファイルとして保存
            if not os.path.exists("../data/"+str(level)):
                os.mkdir("../data/"+str(level))
            save_path = self.rename("../data/"+str(level)+"/mario_trajectory")

            obs,action = self.resize(obs_trajectory,actions_trajectory)

            if obs is None or action is None:
                return
            np.savez_compressed(save_path, observations=obs, actions=action)
            print(f"軌跡データを '{save_path}' に保存しました。")
        else:
            print("データは収集されませんでした。")

    def resize(self,obs,action):
        # ランダムな開始地点から500sampleを保存する
        if(len(obs)< 500):
            print("サンプル数が少ないです")
            return None,None
        num = random.randint(0,len(obs)- 500)
        obs = obs[num:num+500]
        action = action[num:num+500]
        return obs,action

    def rename(self,save_path):
        i = 0
        while True:
            path = save_path + "_" + str(i)
            if not os.path.isfile(path+".npz"):
                break
            else:
                i += 1
        return path
