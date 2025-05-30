import gymnasium as gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import SetPlayingMode
env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
env = SetPlayingMode('algo')(env)

env.reset()

while True:
    obs, reward, terminated,truncated, info = env.step([0,0,0,1,0,1])
    if terminated:
        break

env.close()
