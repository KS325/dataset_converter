
import torch

data = torch.load("/home/sato/data/pt_datasets/KS325/only-drawer/.meta/action_max.pt", map_location="cpu")

print(data)

obs = torch.load("/home/sato/data/pt_datasets/KS325/only-drawer/episode-0/obs.pt", map_location="cpu")

print("Keys:", obs.keys())

print("\n=== Proprio ===")
print(type(obs["action"]))
print(obs["action"].shape)

print("\n=== Camera ===")
for cam_name, img in obs["camera"].items():
    print(cam_name, img.shape)

print("\n=== Mask ===")
print(obs["mask"].shape)

print("\n=== Instruction ===")
print(obs["instruction"])

# import pandas as pd
# from glob import glob

# dfs = pd.read_parquet("/home/sato/data/datasets/KS325/only-drawer/meta/tasks.parquet")
# print(dfs.head())
# print(dfs.columns)
# print(dfs.index)

