from .load import load
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

def play_stage(stage=1,index=0):
    obs_data_tensor = load(stage,index)
    obs_data_tensor = obs_data_tensor.permute(0,3,1,2)
    play(obs_data_tensor)


def play(obs_data_tensor):
    num_frames = 500  # シーケンス内のフレーム数
    channels = 3      # RGB画像
    height = 224       # 画像の高さ
    width = 256        # 画像の幅


    print(f"生成されたobsデータの形状: {obs_data_tensor.shape}")
    print(f"データ型: {obs_data_tensor.dtype}")

# --- 2. アニメーションの準備 ---
    fig, ax = plt.subplots(figsize=(6, 6)) # 表示する図と軸を作成
    ax.axis('off') # 軸を非表示にする（画像表示なので）

# 最初のフレームを表示
# obs_data_tensorは (N, C, H, W) なので、
# 表示のために (H, W, C) に変換 (permute) し、NumPy配列に変換 (.numpy()) 
# RGB画像の場合は (H, W, 3) が必要
    if channels == 3:
        initial_frame = obs_data_tensor[0].permute(1, 2, 0).numpy()
    elif channels == 1:
        initial_frame = obs_data_tensor[0].squeeze(0).numpy() # (1, H, W) -> (H, W)
    else:
        raise ValueError("チャンネル数が1 (グレースケール) または 3 (RGB) の画像を想定しています。")

# imshow で画像オブジェクトを作成
# vmin/vmax はデータの最小値/最大値を指定し、色のマッピングを固定する
# これは、データが0-1の範囲にある場合や、特定の範囲にある場合に重要
    im = ax.imshow(initial_frame, cmap='gray' if channels == 1 else None, vmin=0, vmax=1)


# --- 3. アニメーション更新関数 ---
    def update(frame):
        # 現在のフレームデータを取得
        # obs_data_tensorは既に0-1に正規化されていると仮定
        current_frame = obs_data_tensor[frame]

        # 表示用に形状を変換 (C, H, W) -> (H, W, C) または (H, W)
        if channels == 3:
            display_frame = current_frame.permute(1, 2, 0).numpy()
        elif channels == 1:
            display_frame = current_frame.squeeze(0).numpy()
        else:
            display_frame = current_frame.numpy() # それ以外のチャンネル数の場合（要調整）

        im.set_array(display_frame) # 画像データを更新
        ax.set_title(f"Frame: {frame+1}/{num_frames}") # タイトルを更新
        return [im] # 更新されたオブジェクトをリストで返す

# --- 4. アニメーションの作成と表示 ---
# FuncAnimation(fig, func, frames, interval, blit)
# fig: Figureオブジェクト
# func: 各フレームを更新する関数
# frames: フレームの総数、またはフレームを生成するイテラブル
# interval: 各フレーム間のミリ秒数
# blit: Trueの場合、変更された部分のみを再描画して高速化（複雑な描画では問題を起こすことも）

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50, # 20フレーム/秒 (1000ms / 50ms = 20)
        blit=True,
        repeat=True # 繰り返し再生
    )

    plt.tight_layout() # レイアウトの調整
    plt.show()

# --- 5. アニメーションをファイルとして保存 (任意) ---
# ffmpeg がインストールされている必要があります
# ani.save('observation_sequence.mp4', writer='ffmpeg', fps=20)
# print("アニメーションが 'observation_sequence.mp4' として保存されました。")

if __name__ == "__main__":
    play(sys.argv[1],sys.argv[2])
