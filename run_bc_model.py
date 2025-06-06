from net import MarioBCModel
import gymnasium as gym
import ppaquette_gym_super_mario
import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys

# --- 設定 ---
MODEL_PATH = "mario_bc_model.pth" # 学習済みモデルのパス

# --- メインロジック ---
if __name__ == "__main__":
    # モデルファイルの存在チェック
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: 学習済みモデル '{MODEL_PATH}' が見つかりません。")
        print("先に 'train_bc_model.py' を実行してモデルを作成してください。")
        exit()

    # 環境の作成
    args = sys.argv
    env = gym.make('ppaquette/SuperMarioBros-'+args[1]+'-v0')

    # モデルのロード
    num_actions = 6 # アクションの次元数 (この場合6)
    model = MarioBCModel(num_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # 推論モードに設定 (Dropoutなどが無効になる)

    print(f"学習済みモデル '{MODEL_PATH}' をロードしました。")
    print(f"使用デバイス: {device}")
    print("\n--- モデルによるマリオの実行を開始します ---")
    print("ゲームウィンドウにフォーカスを当てて、マリオの動作を確認してください。")
    print("エピソードが終了するか、ゲームが終了するまで実行されます。")

    # 環境のリセット
    obs, info = env.reset()
    
    # 推論ループ
    episode_reward = 0
    total_frames = 0
    
    while True:
        # 観測値をPyTorchテンソルに変換し、デバイスに移動
        # (H, W, C) -> (1, H, W, C) -> (1, C, H, W)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

        # モデルで行動を予測
        with torch.no_grad(): # 推論時は勾配計算を無効化
            action_logits = model(obs_tensor)
            # 行動をバイナリ（0または1）に変換 (BCEWithLogitsLossを使用した場合)
            action_probs = torch.sigmoid(action_logits)
            predicted_action = (action_probs > 0.5).int().cpu().numpy().squeeze(0) # NumPy配列に変換し、バッチ次元を削除

        # 環境に予測された行動を適用
        # env.step() はリストを期待するので tolist() を呼び出す
        next_obs, reward, terminated, truncated, info = env.step(predicted_action.tolist())

        episode_reward += reward
        total_frames += 1


        # エピソードが終了した場合
        if terminated or truncated:
            print(f"エピソード終了: フレーム数 = {total_frames}, 合計報酬 = {episode_reward:.2f}")
            break # エピソード終了でループを抜ける

        # 次のステップのために観測を更新
        obs = next_obs
        
        # オプション：再生速度の調整 (人間の目で追えるように少し遅延を入れる)
        # 環境が内部でフレームレートを制御している場合、このsleepは不要か、
        # 微調整が必要になることがあります。
        # time.sleep(0.01)

    print("\n--- モデルによる実行が完了しました ---")

    # 環境を閉じる
    env.close()
    print("ゲーム環境が閉じられました。")
