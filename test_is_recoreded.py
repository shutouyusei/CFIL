import gymnasium as gym
import ppaquette_gym_super_mario
from pynput import keyboard # pynput はもはや不要ですが、キー入力部分を削除するため残します
import numpy as np
import threading
import time
import os

# --- 環境と設定 ---
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

# --- データ読み込み ---
LOAD_PATH = "data/mario_trajectory.npz" # 保存した .npz ファイルのパス

# ロードするデータを格納する変数
loaded_observations = None
loaded_actions = None

# データファイルの存在チェック
if not os.path.exists(LOAD_PATH):
    print(f"エラー: データファイル '{LOAD_PATH}' が見つかりません。")
    print("先にデータ収集スクリプトを実行して、データを作成してください。")
    exit() # ファイルがない場合は終了

try:
    print(f"データファイル '{LOAD_PATH}' を読み込み中...")
    loaded_data = np.load(LOAD_PATH)
    loaded_observations = loaded_data['observations']
    loaded_actions = loaded_data['actions']
    print(f"Observations shape: {loaded_observations.shape}")
    print(f"Actions shape: {loaded_actions.shape}")
    print(f"合計 {len(loaded_actions)} フレームのアクションを再生します。")

except Exception as e:
    print(f"データファイルの読み込み中にエラーが発生しました: {e}")
    exit()

# --- メインロジック ---
if __name__ == "__main__":
    print("\n--- 軌跡の再生を開始します ---")

    # 環境をリセット (再生開始)
    obs, info = env.reset()

    # 読み込んだアクションを順に環境に入力
    for frame_idx, action_to_take in enumerate(loaded_actions):
        # アクションを実行 (NumPy配列なので tolist() でリストに変換)
        next_obs, reward, terminated, truncated, info = env.step(action_to_take.tolist())

        # 実行中のフレーム番号を表示
        if (frame_idx + 1) % 100 == 0 or terminated or truncated:
            print(f"再生中: フレーム {frame_idx + 1}/{len(loaded_actions)}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        # エピソードが終了した場合
        if terminated or truncated:
            print(f"エピソードがフレーム {frame_idx + 1} で終了しました。再生を停止します。")
            break

        # 次のステップのために obs を更新 (必要であれば)
        obs = next_obs
        
        # 再生速度の調整 (任意)
        # 環境が内部でフレームレートを制御している場合（ppaquette-gym-super-marioのように）、
        # ここで sleep を入れると二重に遅延が発生する可能性があります。
        # 必要に応じてコメントアウト/調整してください。
        # time.sleep(0.01)

    print("\n--- 軌跡の再生が完了しました ---")

    # 環境を閉じる
    env.close()
    print("ゲーム環境が閉じられました。")
