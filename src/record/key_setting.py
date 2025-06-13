from pynput import keyboard
import numpy as np

class KeySetting:
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

    def start_listen(self):
        self.listener.start()
    def stop_listen(self):
        self.listener.stop()

    def get_actions(self):
        return self.current_actions.tolist()
