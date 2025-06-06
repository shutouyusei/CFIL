import gymnasium as gym
import ppaquette_gym_super_mario

class MarioEnv:
    def create_env(level,tile=0):
        if level == None:
            print("ERROR: no level setting")
            return None
        if tile:
            return gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
        else:
            return gym.make('ppaquette/SuperMarioBros-'+level+'-v0')

