import gymnasium as gym
from .key_setting import KeySetting
from env.mario_env import MarioEnv
from ppaquette_gym_super_mario.wrappers.control import SetPlayingMode
import os

class RecTrajectory:
    def start_rec(self,level):
        self.key_setting = KeySetting()
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

