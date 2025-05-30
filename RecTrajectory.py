import gymnasium as gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import SetPlayingMode
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
env = SetPlayingMode('human')(env)
env.reset()
trajectory = []
while True:
    obs,reward, terminated, truncated, info = env.ste
    if terminated:
        break

