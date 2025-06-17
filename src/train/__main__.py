from .train import *

MODEL_SAVE_PATH = "../data/mario_gail_model.pth"
train = TrainGAIL()

train.train(MODEL_SAVE_PATH)
print(f"モデルを '{MODEL_SAVE_PATH}' に保存しました。")
