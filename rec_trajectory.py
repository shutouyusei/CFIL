import gymnasium as gym
import numpy as np
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import SetPlayingMode
from pynput import keyboard

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

env.reset()
ACTION_MAPPING = {
    keyboard.KeyCode(char='w'):0, #right
    keyboard.KeyCode(char='a'):1, #left
    keyboard.KeyCode(char='s'):2, #down
    keyboard.KeyCode(char='d'):3, #right
    keyboard.KeyCode(char='k'):4, #a
    keyboard.KeyCode(char='j'):5, #b
}

current_actions = np.array([0,0,0,0,0,0],dtype=np.int8) 
# キーイベントハンドラ
def on_press(key):
    # ACTION_MAPPING に直接キーオブジェクトがある場合
    if key in ACTION_MAPPING:
        current_actions[ACTION_MAPPING[key]] = 1
    elif isinstance(key, keyboard.KeyCode) and key.char:
        # char属性を持つキー（文字キー）の場合、charで比較する
        for pyn_key_obj, idx in ACTION_MAPPING.items():
            if isinstance(pyn_key_obj, keyboard.KeyCode) and pyn_key_obj.char == key.char:
                current_actions[idx] = 1
                break

def on_release(key):
    if key in ACTION_MAPPING:
        current_actions[ACTION_MAPPING[key]] = 0
    elif isinstance(key, keyboard.KeyCode) and key.char:
        for pyn_key_obj, idx in ACTION_MAPPING.items():
            if isinstance(pyn_key_obj, keyboard.KeyCode) and pyn_key_obj.char == key.char:
                current_actions[idx] = 0
                break

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

all_observations = []
all_actions = []
while True:
    obs, reward, terminated, truncated, info = env.step(current_actions.tolist())
    if terminated:
        break
    all_observations.append(obs)
    all_actions.append(current_actions.tolist())

listener.stop()
env.close()

if all_observations and all_actions:
    print(f"\nCollected {len(all_observations)} frames of data.")
    
    # リストをNumPy配列に変換
    obs_trajectory = np.array(all_observations)
    actions_trajectory = np.array(all_actions)

    # データを .npz ファイルとして保存
    save_path = "data/mario_trajectory.npz"

    np.savez_compressed(save_path, observations=obs_trajectory, actions=actions_trajectory)
    print(f"軌跡データを '{save_path}' に保存しました。")
    print(f"Observations shape: {obs_trajectory.shape}")
    print(f"Actions shape: {actions_trajectory.shape}")
else:
    print("データは収集されませんでした。")
