import gymnasium as gym
import numpy as np
from mario_env import MarioEnv
from ppaquette_gym_super_mario.wrappers.control import SetPlayingMode
from pynput import keyboard
import random
import os
import sys

class RecTrajectory:
    def __init__(self):
        self.ACTION_MAPPING = {
            keyboard.KeyCode(char='w'):0, # up
            keyboard.KeyCode(char='a'):1, # left
            keyboard.KeyCode(char='s'):2, # down
            keyboard.KeyCode(char='d'):3, # right
            keyboard.KeyCode(char='k'):4, # a
            keyboard.KeyCode(char='j'):5, # b
        }
        self.current_actions = np.array([0,0,0,0,0,0],dtype=np.int8) 
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self,key):
        # ACTION_MAPPING に直接キーオブジェクトがある場合
        if key in self.ACTION_MAPPING:
            self.current_actions[self.ACTION_MAPPING[key]] = 1
        elif isinstance(key, keyboard.KeyCode) and key.char:
            # char属性を持つキー（文字キー）の場合、charで比較する
            for pyn_key_obj, idx in ACTION_MAPPING.items():
                if isinstance(pyn_key_obj, keyboard.KeyCode) and pyn_key_obj.char == key.char:
                    self.current_actions[idx] = 1
                    break

    def on_release(self,key):
        if key in self.ACTION_MAPPING:
            self.current_actions[self.ACTION_MAPPING[key]] = 0
        elif isinstance(key, keyboard.KeyCode) and key.char:
            for pyn_key_obj, idx in self.ACTION_MAPPING.items():
                if isinstance(pyn_key_obj, keyboard.KeyCode) and pyn_key_obj.char == key.char:
                    self.current_actions[idx] = 0
                    break

    def start_rec(self,level):
        self.env = MarioEnv.create_env(level)
        self.listener.start()
        self.env.reset()
        all_observations = []
        all_actions = []
        while True:
            obs, reward, terminated, truncated, info = self.env.step(self.current_actions.tolist())
            if terminated:
                break
            all_observations.append(obs)
            all_actions.append(self.current_actions.tolist())
        self.listener.stop()
        self.env.close()
        return all_observations, all_actions

    def save_trajectory(self,level,all_observations,all_actions):
        if all_observations and all_actions:
            print(f"\nCollected {len(all_observations)} frames of data.")
            # リストをNumPy配列に変換
            obs_trajectory = np.array(all_observations)
            actions_trajectory = np.array(all_actions)

            # データを .npz ファイルとして保存
            if not os.path.exists("data/"+str(level)):
                os.mkdir("data/"+str(level))
            save_path = self.rename("data/"+str(level)+"/mario_trajectory")
            # ランダムな開始地点から500sampleを保存する
            if(len(obs_trajectory)< 500):
                print("サンプル数が少ないです")
                return
            num = random.randint(0,len(obs_trajectory)- 500)
            obs_trajectory= obs_trajectory[num:num+500]
            actions_trajectory = actions_trajectory[num:num+500]

            np.savez_compressed(save_path, observations=obs_trajectory, actions=actions_trajectory)
            print(f"軌跡データを '{save_path}' に保存しました。")
        else:
            print("データは収集されませんでした。")

    def rename(self,save_path):
        i = 0
        while True:
            path = save_path + "_" + str(i)
            if not os.path.isfile(path+".npz"):
                break
            else:
                i += 1
        return path

    def main(self,level):
        obs,actions = self.start_rec(level)
        self.save_trajectory(level,obs,actions)

if __name__ == "__main__":
    args = sys.argv
    RecTrajectory().main(args[1])
