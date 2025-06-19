import numpy as np
import random
import os

class SaveTrajectory:
    def save(self,level,all_observations,all_actions,path):
        if all_observations and all_actions:
            print(f"\nCollected {len(all_observations)} frames of data.")
            obs_trajectory,actions_trajectory = self.__get_np_trajectory(all_observations,all_actions)

            if not os.path.exists(path):
                exit("path is not exist")

            if path is None:
                print("No path")
                return
            save_path = self.__rename(path + "/" + str(level) + "/mario_trajectory")
            print(save_path)

            if(len(obs_trajectory)< 500):
                print("サンプル数が少ないです")
                return
            obs,action = self.__resize(obs_trajectory,actions_trajectory)

            np.savez_compressed(save_path, observations=obs, actions=action)
            print(f"軌跡データを '{save_path}' に保存しました。")
        else:
            print("データは収集されませんでした。")

    def __get_np_trajectory(self, obs, action):
        obs_trajectory = np.array(obs)
        actions_trajectory = np.array(action)
        return obs_trajectory,actions_trajectory

    def __resize(self,obs,action):
        num = random.randint(0,len(obs)- 500)
        obs = obs[num:num+500]
        action = action[num:num+500]
        return obs,action

    def __rename(self,save_path):
        i = 0
        while True:
            path = save_path + "_" + str(i)
            if not os.path.isfile(path+".npz"):
                break
            else:
                i += 1
        return path
